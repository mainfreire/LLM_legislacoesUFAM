import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = "/content/logs"

event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

tags = event_acc.Tags()
print("Tags disponíveis:", tags)

scalars = {tag: event_acc.Scalars(tag) for tag in tags['scalars']}
print("Scalars disponíveis:", scalars.keys())

train_loss_data = scalars.get('train/train_loss', [])
epoch_data = scalars.get('train/epoch', [])

print(f"Train Loss Data: {len(train_loss_data)} entries")
print(f"Epoch Data: {len(epoch_data)} entries")
steps_train_loss = [entry.step for entry in train_loss_data]
values_train_loss = [entry.value for entry in train_loss_data]

steps_epoch = [entry.step for entry in epoch_data]
values_epoch = [entry.value for entry in epoch_data]
print("Steps Train Loss:", steps_train_loss)
print("Values Train Loss:", values_train_loss)
print("Steps Epoch:", steps_epoch)
print("Values Epoch:", values_epoch)

plt.figure(figsize=(14, 6))

# grafico de perda de treinamento
plt.subplot(1, 2, 1)
plt.plot(steps_train_loss, values_train_loss, label='Train Loss', marker='o')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# grafico de época
plt.subplot(1, 2, 2)
plt.plot(steps_epoch, values_epoch, label='Epoch', marker='o')
plt.xlabel('Step')
plt.ylabel('Epoch')
plt.title('Epoch Progress')
plt.legend()

if len(steps_train_loss) > 0:
    plt.xlim(min(steps_train_loss) - 1, max(steps_train_loss) + 1)
    plt.ylim(min(values_train_loss) - 0.1, max(values_train_loss) + 0.1)
if len(steps_epoch) > 0:
    plt.xlim(min(steps_epoch) - 1, max(steps_epoch) + 1)
    plt.ylim(min(values_epoch) - 0.1, max(values_epoch) + 0.1)

# Mostrar gráficos
plt.show()
