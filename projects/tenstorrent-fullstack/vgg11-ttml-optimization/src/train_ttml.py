                                                   

import random
import time
from pathlib import Path

import numpy as np
import torch
import ttnn
import ttml

from dataset import classification_loaders
from vgg_ttml import FrozenVGG11, VGGClassifier


          
DATA_ROOT = "~/datasets/oxford_pet"
SAVE_DIR = Path("run/ttml")
DOWNLOAD = False                       
WEIGHTS = None                                                 
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 512
NUM_CLASSES = 37
SEED = 42
DEVICE_ID = 0
WARMUP_BATCHES = 10                         


def run_epoch(backbone, classifier, loader, optimizer, context, training):
                                            
    if training:
        classifier.train()
        context.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    else:
        classifier.eval()
        context.set_gradient_mode(ttml.autograd.GradMode.DISABLED)

    loss_sum = 0.0
    correct = 0
    sample_count = 0

    model_time_sum = 0.0

    for step, (images, labels) in enumerate(loader, start=1):
                                             
        targets = ttml.autograd.Tensor.from_numpy(
            labels.numpy().astype(np.uint32).reshape(1, -1),
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
        )
        targets.set_requires_grad(False)

        if training:
            optimizer.zero_grad()                  

                                           
        if step > WARMUP_BATCHES:
            ttnn.synchronize_device(context.get_device())
            model_start = time.perf_counter()

        features = backbone(images)                           
        outputs = classifier(features)                    

        if step > WARMUP_BATCHES:
            ttnn.synchronize_device(context.get_device())
            model_time_sum += time.perf_counter() - model_start

        loss = ttml.ops.loss.cross_entropy_loss(outputs, targets)

        loss_value = float(loss.to_numpy(ttnn.DataType.FLOAT32).item())
        if not np.isfinite(loss_value):
            raise FloatingPointError("loss가 유효한 숫자가 아닙니다.")

        if training:
            loss.backward(False)                
            optimizer.step()                  

                              
        scores = outputs.to_numpy(ttnn.DataType.FLOAT32)
        predictions = scores.reshape(len(labels), NUM_CLASSES).argmax(axis=1)
        correct += int((predictions == labels.numpy()).sum())
        loss_sum += loss_value * len(labels)
        sample_count += len(labels)
        running_accuracy = correct / sample_count
        context.reset_graph()                     

                                               
        if step == WARMUP_BATCHES:
            ttnn.synchronize_device(context.get_device())
            print(f"배치 {step}/{len(loader)} | loss {loss_value:.4f} | "
                  f"acc {running_accuracy:.2%} | 워밍업 완료", flush=True)
        elif step > WARMUP_BATCHES and step % 10 == 0:
            measured_batches = step - WARMUP_BATCHES
            average_batch_ms = model_time_sum / measured_batches * 1000
            print(f"배치 {step}/{len(loader)} | loss {loss_value:.4f} | "
                  f"acc {running_accuracy:.2%} | "
                  f"평균 {average_batch_ms:.2f} ms/batch", flush=True)

    if sample_count == 0:
        raise ValueError("데이터가 없습니다.")
    if step <= WARMUP_BATCHES:
        raise ValueError(f"속도 측정에는 {WARMUP_BATCHES + 1}개 이상의 배치가 필요합니다.")

    measured_batches = step - WARMUP_BATCHES
    average_batch_ms = model_time_sum / measured_batches * 1000
    return loss_sum / sample_count, correct / sample_count, average_batch_ms


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

                 
    train_loader, val_loader, class_names = classification_loaders(
        DATA_ROOT, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        seed=SEED, download=DOWNLOAD,
    )
    if set(class_names) != set(range(NUM_CLASSES)):
        raise ValueError("Oxford Pet 37개 품종 데이터가 필요합니다.")

              
    context = ttml.autograd.AutoContext.get_instance()
    context.set_seed(SEED)
    context.open_device(
        device_ids=[DEVICE_ID],
        l1_small_size=8 * 1024,                  
    )

    try:
                                                 
        backbone = FrozenVGG11(context.get_device(), IMAGE_SIZE, WEIGHTS)
        classifier = VGGClassifier(backbone.out_features, NUM_CLASSES, HIDDEN_SIZE)
        optimizer_config = ttml.optimizers.AdamWConfig.make(
            lr=LEARNING_RATE, beta1=0.9, beta2=0.999,
            epsilon=1e-8, weight_decay=0.0001,
        )
        optimizer = ttml.optimizers.AdamW(classifier.parameters(), optimizer_config)

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state, SAVE_DIR / "backbone.pt")
        best_accuracy = -1.0

                             
        ttnn.synchronize_device(context.get_device())
        total_start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS} — 학습", flush=True)
            train_loss, train_acc, train_batch_ms = run_epoch(
                backbone, classifier, train_loader, optimizer, context, training=True,
            )
            print("검증", flush=True)
            val_loss, val_acc, val_batch_ms = run_epoch(
                backbone, classifier, val_loader, optimizer, context, training=False,
            )
            print(f"학습: loss {train_loss:.4f}, 정확도 {train_acc:.2%}")
            print(f"검증: loss {val_loss:.4f}, 정확도 {val_acc:.2%}")
            print(f"배치당 평균: 학습 {train_batch_ms:.2f} ms/batch | "
                  f"검증 {val_batch_ms:.2f} ms/batch")

            checkpoint = {
                "classifier": classifier.cpu_state_dict(),
                "image_size": IMAGE_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_classes": NUM_CLASSES,
                "class_names": class_names,
                "epoch": epoch,
                "val_accuracy": val_acc,
            }
            torch.save(checkpoint, SAVE_DIR / "last.pt")
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(checkpoint, SAVE_DIR / "best.pt")
                print("최고 정확도 모델 저장")
        total_seconds = time.perf_counter() - total_start
        print(f"\n전체 소요 시간: {total_seconds / 60:.2f}분 "
              f"({total_seconds:.1f}초, 학습·검증·저장 포함 / 초기 준비 제외)")
    finally:
                           
        context.reset_graph()
        context.close_device()


if __name__ == "__main__":
    main()
