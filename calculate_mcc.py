from fairseq.models.roberta import RobertaModel
from sklearn.metrics import matthews_corrcoef

roberta = RobertaModel.from_pretrained(
    'bitfit/large/CoLA/30-15-16-4e-4-1',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/new_home/zhuocheng/FastBERT/examples/roberta/glue/CoLA-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
y_true, y_pred = [], []
with open('/new_home/zhuocheng/dev.tsv') as fin:
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent, target = tokens[-1], tokens[1]
        tokens = roberta.encode(sent)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        y_true.append(int(target))
        y_pred.append(int(prediction_label))
        nsamples += 1
print('matthews_corrcoef: ' + str(matthews_corrcoef(y_true, y_pred)))
print('ncorrect / nsamples = ' + str(ncorrect) + ' / ' + str(nsamples))
print('| Accuracy: ', float(ncorrect)/float(nsamples))