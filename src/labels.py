import csv

def load(labelfile):
  labels = []
  with open(labelfile) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      labels.append(row[0])
  return labels

if __name__ == "__main__":
    filename = '../data/labels.txt'
    labels = load(filename)
    print(labels)