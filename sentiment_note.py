import csv
import os

results_dir=os.path.join(os.getcwd(),'results')
results_filenames=os.listdir(results_dir)
print('results file names:',results_filenames)

id=[]
results=[]
num_cls=len(results_filenames)
scores=[]
first_flag=True
for result_name in results_filenames:
    score=result_name[0:-4]
    scores.append(float(score))
    classifications=[]
    result_path=os.path.join(results_dir,result_name)
    f=open(result_path,'r')
    csv_reader = csv.reader(f, delimiter=',')
    head_row = next(csv_reader)
    print('head row:',head_row)
    for line in csv_reader:
        classifications.append(float(line[1]))
        if first_flag:
            id.append(line[0])
    first_flag = False
    results.append(classifications)            

print('scores:',scores)
num=len(id)
print('num of examples:',num)
note_pred=[]
for i in range(num):
    pos=0.0
    for j,classifications in enumerate(results):
        pos=pos+(classifications[i]*scores[j])
    note_pred.append(1 if (pos/sum(scores))>0.5 else 0)
    #note_pred.append(1 if (pos / sum(num)) > 0.5 else 0)

#write
result=zip(id,note_pred)

f=open('result.csv', 'w',newline='')
csv_writer=csv.writer(f,delimiter=',')
csv_writer.writerow(['id','sentiment'])
csv_writer.writerows(result)
f.close()







