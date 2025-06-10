
from feature_extraction import *
from models import *
from pipeline import Pipeline

channel = [1,2,3,4,6,11,13,17,19,20,21,25,29,31] #14 Channels chosen to fit Emotiv Epoch+
band = [4,8,12,16,25,45] #5 bands
window_size = 256 #Averaging band power of 2 sec
step_size = 16 #Each 0.125 sec update once
sample_rate = 128 #Sampling rate of 128 Hz
subjectList = ['01','02','03']
input_path = "E:\DEAP\data_preprocessed_python\s"
output_path = "E:\DEAP\data_preprocessed_python\FFT_npy_data\s"
num_classes = 10



def main():
    for sub in subjectList:
        feature_extraction(sub, channel, band, window_size, step_size, sample_rate, input_path, output_path, is_3D= True)
    X_train, y_train, X_test, y_test = load_data(subjectList, output_path,label_index=3, samples_per_trial=464)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_3D(num_classes=num_classes).to(device)
    pipeline = Pipeline(model, X_train, y_train, X_test, y_test, mode = '3D')
    pipeline.train()

if __name__ == "__main__":
    main()