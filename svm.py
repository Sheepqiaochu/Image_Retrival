import numpy as np
from sklearn import svm
from sklearn.datasets  import load_digits
from sklearn.model_selection  import train_test_split
import _pickle as pickle
from datasets import CUB_200_2011, Stanford_Dog
from utils import *
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM')
    parser.add_argument('--k', type=str, default='rbf',help='kernel function')
    parser.add_argument('--c', type=float, default=1.0,
                        help='relaxation')

    args = parser.parse_args()
    train_dataset = Stanford_Dog('datasets/Stanford_Dogs', True)
    test_dataset = Stanford_Dog('datasets/Stanford_Dogs', False)

    # build train&test data
    train_x = []
    train_y = []
    for i in range(len(train_dataset)):
        img, target = train_dataset[i]
        train_x.append(img.view(-1).numpy())
        train_y.append(target)

    test_x = [] 
    test_y = []
    for i in range(len(test_dataset)):
        img, target = test_dataset[i]
        test_x.append(img.view(-1).numpy())
        test_y.append(target)

    print('-------开始加载模型--------')
    model = svm.SVC(gamma='scale', C=args.c, decision_function_shape='ovr', kernel=args.k)
    model.fit(train_x, train_y)

    z = model.predict(test_x)
    print('准确率:',np.sum(z==test_y)/z.size)
    z_t = torch.from_numpy(z)

    with open('./model.pkl','wb') as file:
        pickle.dump(model,file)