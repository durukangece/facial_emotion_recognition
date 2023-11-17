import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from DataLoader import FerDataset
# from Cnn_Model import TransferLearningModel
from Transfer_Learning_Model import TransferLearningModel
from Cnn_Model import SimpleCNN
import os
import datetime
import matplotlib.pyplot as plt

try: # try-except bloğu, olası hataların yönetilmesi için kullanılır. Eğitim sürecinde ortaya çıkabilecek herhangi bir hata yakalanır ve ekrana basılır.
    # For train and validation
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Hyperparameters
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.0001
    num_classes = 7

    # Data paths
    train_data_dir = "Emotion Recognition/train"
    val_data_dir = "Emotion Recognition/test"

    # Data transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Görüntüyü yeniden boyutlandırma
        transforms.ToTensor(),         # Görüntüyü PyTorch tensor formatına çevirme
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Bu, görüntülerin her bir pikselinin ortalama değerini 0.5 yapar ve standart sapmasını 0.5 yapar. 
    ])                                                                  # Bu normalizasyon işlemi, modelin daha hızlı ve daha kararlı bir şekilde öğrenmesine yardımcı olur.

    # Create training dataset and loader
    train_dataset = FerDataset(train_data_dir, transform=data_transform, augment=True) # FerDataset sınıfını kullanarak eğitim veri setini oluşturur. 
                                                                                       # Bu, veri setinin yolu (train_data_dir), veri transformasyonları (data_transform) ve 
                                                                                       # veri artırma (data augmentation) işlemlerini içerir (augment=True).
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # DataLoader sınıfını kullanarak eğitim veri setini yükler. train_dataset parametresi, 
                                                                                       # kullanılacak veri setini belirtir. batch_size parametresi, her bir eğitim iterasyonunda kullanılacak 
                                                                                       # mini-batch boyutunu belirtir. shuffle=True parametresi, her epoch öncesinde veri setinin karıştırılmasını sağlar,
                                                                                       # bu da modelin daha iyi öğrenmesine yardımcı olur.
    
    # Create validation dataset and loader
    val_dataset = FerDataset(val_data_dir, transform=data_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Genellikle, model eğitimi sırasında veri artırma (data augmentation) yalnızca eğitim veri seti için yapılır.
    # Bu, modelin gerçek dünya verilerini işlerken nasıl performans göstereceğini daha doğru bir şekilde değerlendirmemizi sağlar.

    print('Train images:', len(train_dataloader.dataset))
    print('Test images:', len(val_dataloader.dataset))
    print('*' * 50)

    # Create the model
    model = SimpleCNN(num_classes)
    # model = TransferLearningModel(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()   # CrossEntropyLoss, genellikle multi-class sınıflandırma problemlerinde kullanılır.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Hiperparametre ayarlamalarının az olması sebebiyle kullanımı daha kolaydır. Adam, genellikle hızlı ve etkili bir şekilde eğitim yapmak için tercih edilen optimizasyon algoritmalarından biridir.

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0 
        train_correct = 0 
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_dataloader): # train_dataloader üzerinde döngü oluşturur. enumerate fonksiyonu ile her bir batch için bir indis (batch_idx) ve içindeki veri tuple'ını (images, labels) döndürür.
            images, labels = images.to(device), labels.to(device) # PyTorch'ta tensor'ları belirli bir cihaza göndermek için kullanılan bir yöntemdir. 
            optimizer.zero_grad() # Her bir batch başlangıcında, gradyanları sıfırlamak için kullanılır
            outputs = model(images) 
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            loss.backward()
            optimizer.step()
            train_accuracy = 100 * train_correct / train_total 
            print(f"Train - Epoch [{epoch+1}/{num_epochs}], Iteration: {batch_idx + 1}/{len(train_dataloader)}, Train Loss: {train_loss:.4f}", f"Train Accuracy: {train_accuracy:.2f}", end='\r')
           
        train_loss /= len(train_dataloader.dataset)
        train_accuracy = 100 * train_correct / train_total

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss_item = criterion(outputs, labels)
                val_loss += val_loss_item.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                print(f"Validation - Epoch [{epoch+1}/{num_epochs}], Iteration: {batch_idx + 1}/{len(val_dataloader)}, Validation Loss: {val_loss_item.item():.4f}",f"Validation Accuracy: {(100*val_correct/val_total):.2f}", end='\r')

            print()
        
        val_loss /= len(val_dataloader.dataset)
        val_accuracy = 100 * val_correct / val_total

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if epoch != num_epochs - 1:
            print('-' * 50) 
        else:
            print('*' * 50)

    # Save the model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = os.path.join("results/models", f"model_{timestamp}")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Modeli kaydetme
    model_save_path = os.path.join(model_save_dir, f"model_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)

    epoch_numbers = list(range(1, num_epochs + 1))

    # loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_numbers, train_losses, label='Training Loss')
    plt.plot(epoch_numbers, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epoch_numbers) # This ensures that the x-axis ticks correspond to the epochs
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, f"loss_{timestamp}.png"))

    # accuracy graph
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_numbers, train_accuracies, label='Training Accuracy')
    plt.plot(epoch_numbers, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(epoch_numbers) # This ensures that the x-axis ticks correspond to the epochs
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, f"accuracy_{timestamp}.png"))


    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    torch.save(model.state_dict(), model_save_path)

except Exception as e:
    print(f"An error occurred: {e}")
