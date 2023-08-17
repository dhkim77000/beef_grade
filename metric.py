from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

def get_metric(metric_name):
    if metric_name == 'accuracy':
        return accuracy_score
    
    elif metric_name == 'f1macro':
        metric = F1Score(average='macro')
        return metric.get_score

class F1Score:
    
    def __init__(self, average):
        self.average = average
        
    def get_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average=self.average)

def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i-j)**2.0/(n-1)**2)
            oij = c_matrix[i,j]
            eij = c_matrix[i,:].sum() * c_matrix[:,j].sum()/c_matrix.sum()
            numer += wij*oij
            denom += wij*eij
    return 1.0 - numer/denom

def get_ck_score(preds, labels, unique_labels):

    c_matrix = confusion_matrix(labels, preds, unique_labels)
    kappa = quadratic_weighted_kappa(c_matrix)

    ck_score = cohen_kappa_score(preds, labels, weights='quadratic', labels=unique_labels, sample_weight=None)

    return ck_score



