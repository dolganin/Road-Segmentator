from utilities.create_objects import model_create
from utilities.load_and_split import  splitting, dataload, transformers
from train_test.test import test_model
from train_test.train import run_model
from imports import epochs, output_model_dir, output_graphics_dir, torch
from utilities.graphics import loss_graphics
from utilities.dataset import SegmentationDataset

def main():
    
    train_transform, test_transform = transformers()
    
    nails = SegmentationDataset(train_transform, test_transform)
    
    (train, test) = splitting(nails)
    
    train_loader, test_loader = dataload(train, test)

    model, optimizer, loss = model_create()
    
    if __name__ == "__main__":
        for i in range(epochs):
            print("Epoch #"+(str(i)))
            run_model(model, optimizer, train_loader, test_loader, loss)
    
        print("Finished training for whole epochs")
    
    print("Started testing model")
    
    test_model(model, test_loader)
    
    torch.save(model.state_dict(), output_model_dir+"model.pt")
    loss_graphics(output_graphics_dir)
    
main()
        
