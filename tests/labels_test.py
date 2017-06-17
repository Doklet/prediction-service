import pytest
import labels

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(here)
sys.path.insert(0, here)

def test_answer():
    filename = '../data/labels.txt'
    labels.load(filename)
