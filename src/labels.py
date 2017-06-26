import csv

def load(labelfile):
  labels = []
  with open(labelfile) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      labels.append(row[0])
  return labels

if __name__ == "__main__":
    filename = '/var/lib/skyraid/models/1/33c066e5-c713-4a1b-83dd-d298b11d3cef/v1/labels.txt'
    # filename = '../data/labels.txt'
    labels = load(filename)
    print(labels)