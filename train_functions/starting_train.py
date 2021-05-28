import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import sys
<<<<<<< HEAD
sys.path.insert(1, '/Users/vinayshukla/Desktop/cassav-project/projects-skeleton-code/networks')
=======
sys.path.insert(1, '/content/projects-skeleton-code/networks')
>>>>>>> a5b3c73b55f07b1492e2c15ca234576f515625db
from StartingNetwork import CNN

def initializationFunction(training_dataset, val_dataset):
    hyperparameters = {'epochs': 20, 'batch_size': 16}
    n_eval = 100

    summary_path = "./log"

    model = CNN(3, 5)

    #Temporary


    starting_train(training_dataset, training_dataset, model, hyperparameters, n_eval, summary_path)



def starting_train(
    train_dataset, val_dataset, model , hyperparameters, n_eval, summary_path
):
    """
    Trains and evaluates a model.
    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    #train_loader = torch.utils.data.DataLoader(
     #   train_dataset, batch_size=batch_size, shuffle=True
    #)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    if summary_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    for epoch in range(epochs):
        for batch in train_dataset:
            images, labels = batch
            print("Images: " , images.shape)

            #images = images.to(device)
            #labels = labels.to(device)

            #probably need to reshape this is its not 784
            #images = torch.reshape(images, (-1,23040000))
            outputs = model.forward(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                print("Here")
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                #evaluate(train_dataset, model, loss_fn)

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                #evaluate(val_loader, model, loss_fn)

            step += 1

        print('Epoch:', epoch, 'Loss:', loss)

    evaluate(train_dataset, model, loss_fn)

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.
    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]
    Example output:
        0.75
    """
    n_correct = 0
    total = 0
    predictions = torch.argmax(outputs, dim=1)


    n_correct += (predictions == labels).int().sum()
    total += len(predictions)
    
    return (n_correct / total).item()


def evaluate(loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    step = 0

    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter(log_dir='./logs')
    total = 0
    epochs = 0
    for epoch in range(loader.batch_size):
        with torch.no_grad():
            curr_batch = 0
            for batch in loader:
                images, labels = batch
                
                #images = images.to(device)
                #labels = labels.to(device)
                
                output = model.forward(images)
                accuracy = compute_accuracy(output, labels)
                curr_batch += accuracy
                loss = loss_fn(output, labels)
                
                
                writer.add_scalar('train_loss', loss, global_step=step)
                writer.add_scalar('accuracy', accuracy, global_step=step)

                step += 1
        total+=(curr_batch/loader.batch_size)
        epochs+=1
        print("Hi")
    print("Calculated ", (total/epochs)*100)