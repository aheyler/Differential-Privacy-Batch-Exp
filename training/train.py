import torch

def train(device, net, optimizer, scheduler, criterion, train_loader, test_loader, epochs):
    model = net.to(device)
    total_step = len(train_loader)
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    for epoch in range(epochs):
        correct = 0
        total = 0
        flag = 0
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to configured device
            images = images.to(device)
            labels = labels.to(device)
            #Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            optimizer.step()
            if (i+1) % 1000 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))

        train_loss_values.append(running_loss)
        train_error.append(100-100*correct/total)
        
        val_acc, avg_val_loss = eval(device, model, test_loader)

        print('Accuracy of the network on the test images: {} %'.format(100 * val_acc))
        val_error.append(100-100*val_acc)
        val_loss_values.append(avg_val_loss)
        scheduler.step()
    return val_error, val_loss_values, train_error, train_loss_values


def eval(device, model, criterion, test_loader): 
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(labels, outputs).item()
    
    accuracy = correct / total
    avg_val_loss = loss / total
    return accuracy, avg_val_loss