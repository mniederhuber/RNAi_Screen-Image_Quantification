import os
import csv

with open('sample_sheet.tsv', 'w') as f:
  fwriter = csv.writer(f, delimiter = '\t')
  fwriter.writerow(['line','date','fp'])
  for root, dirs, files in os.walk('.'):
    for ff in files:
      if '.lif' in ff and '.lifext' not in ff and 'E93' not in ff:
        if '_' in root:
          fp = os.path.abspath(os.path.join(root,ff))
          date = root.split('/')[1].split('_')[0]
          line = ff.split('_')[0]
          row = [line,date,fp]
          fwriter.writerow(row)
        elif '_' in ff:
          fp = os.path.abspath(os.path.join(root,ff))
          date = root.split('/')[1]
          line = ff.split('_')[0]
          row = [line,date,fp]
          fwriter.writerow(row)
        else:
          fp = os.path.abspath(os.path.join(root,ff))
          date = root.split('/')[1]
          line = ff.split('.')[0]
          row = [line,date,fp]
          fwriter.writerow(row)
