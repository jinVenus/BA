from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from utils.plot_confusion_matrix import plotConfusionMatrix

# move to jupyter
class ComputeEvalution:
    def evalution(score, t, label, load_dict, trainPoint, validationPoint):
        result_test = [0] * (len(score))
        for i in range(len(score)):
            if score[i] > t:
                result_test[i] = 1


        print("data : 7000-20000")
        auc = roc_auc_score(label[validationPoint:], result_test)
        # f1_score = f1_score(label[7000:], result_test)
        recall = recall_score(label[validationPoint:], result_test)
        con = confusion_matrix(label[validationPoint:], result_test)
        print("Auc : " + str(auc))
        print("F1_Score : " + str(f1_score))
        print("recall : " + str(recall))
        print("confusion_matrix : " + str(con))
        plotConfusionMatrix.plot_confusion_matrix(con, name="confusionMatrixPart1", load_dict=load_dict)


        print("data : 7000-10000")
        test = result_test[0:validationPoint-trainPoint]
        auc = roc_auc_score(label[trainPoint:validationPoint], test)
        # f1_score = f1_score(label[7000:10000], test)
        recall = recall_score(label[trainPoint:validationPoint], test)
        con = confusion_matrix(label[trainPoint:validationPoint], test)
        print("Auc : " + str(auc))
        print("F1_Score : " + str(f1_score))
        print("recall : " + str(recall))
        print("confusion_matrix : " + str(con))
        plotConfusionMatrix.plot_confusion_matrix(con, name="confusionMatrixPart2", load_dict=load_dict)


        print("data : 10000-20000")
        auc = roc_auc_score(label[10000:], result_test[3000:len(score)])
        # f1_score = f1_score(label[10000:], result_test[3000:len(score)])
        recall = recall_score(label[10000:], result_test[3000:len(score)])
        con = confusion_matrix(label[10000:], result_test[3000:len(score)])
        print("Auc : " + str(auc))
        print("F1_Score : " + str(f1_score))
        print("recall : " + str(recall))
        print("confusion_matrix : " + str(con))

        plotConfusionMatrix.plot_confusion_matrix(con, name="confusionMatrixPart3", load_dict=load_dict)
