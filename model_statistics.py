#!/usr/bin/env python3

from utils import *
from network import *

def get_model_statistics(pair_fname=PAIRS_FILE, model_fname=MODEL_FNAME, threshold=THRESHOLD):
    '''
    Generate model statistics for a given model stored in checkpoint

    Parameters: pair_file : filename for pairs data
                checkpoint_fname : filename for checkpoint
                threshold: int
                           threshold for converting probabilities to classes
    '''

    pairs_dataset = VoxCelebDataset(pair_fname, train=True)   #FIXME
    test_model, _, _ = load_saved_model(model_fname)
    test_model = test_model.to(device)
    all_losses, true_labels, pred_proba  = [], [], []
    with torch.no_grad():
        for i, data  in enumerate(pairs_dataset):
            mfcc1, mfcc2, label = data['spec1'], data['spec2'], data['label']
            mfcc1 = torch.tensor(mfcc1)
            mfcc2 = torch.tensor(mfcc2)
            mfcc1 = mfcc1.view(test_batch_size, 1, mfcc1.shape[1], mfcc1.shape[2])
            mfcc2 = mfcc2.view(test_batch_size, 1, mfcc2.shape[1], mfcc2.shape[2])
            mfcc1 = mfcc1.to(device)
            mfcc2 = mfcc2.to(device)
            true_labels.append(label)
            output1, output2 = test_model(mfcc1, mfcc2)
            euclidean_distance = F.pairwise_distance(output1, output2)
#             euclidean_distance.item()
#             pair_similarity = cosine_similarity(output1.detach().cpu().numpy(), output2.detach().cpu().numpy())
            pred_proba.append(euclidean_distance.item())
            if i > 1000:
                break

    # positive class = 0, negative = 1
    pred_labels = np.where(np.asarray(pred_proba) <= threshold, 1, 0)
    true_labels = np.asarray(true_labels)
    precision, recall, fscore, _ = score(true_labels, pred_labels, pos_label=0, average='binary')

    print("Precision {}".format(precision))
    print("Recall {}". format(recall))
    print("F1 Score {}". format(fscore))

    with open('model_stats.txt', "a") as stats_file:
        stats_file.write("Model Name: " + str(model_fname) + "\n")
        stats_file.write("Threshold: " + str(threshold) + "\n")
        stats_file.write("Precision: " + str(precision) + "\n")
        stats_file.write("Recall: " + str(recall) + "\n")
        stats_file.write("F1 Score: " + str(fscore) + "\n\n")

    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_proba, pos_label=0)

    #Build ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    fname = "roc_" + time.strftime("%Y%m%d-%H%M%S") + "_threshold_" + str(threshold) + \
            "_model_fname_" + "_".join(model_fname.split('.')[:-2]) + ".png"
    plt.savefig(fname)

    # Print AUC
    auc = metrics.auc(fpr, tpr)
    print('AUC: {}'.format(auc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Model Statistics')
    parser.add_argument('-tr', help='Threshold for calculating class labels', default=THRESHOLD, type=float)
    parser.add_argument('-f', help='Model fname', default=MODEL_FNAME, type=str)
    args = parser.parse_args()
    test_batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    get_model_statistics(threshold=args.tr, model_fname=args.f)
