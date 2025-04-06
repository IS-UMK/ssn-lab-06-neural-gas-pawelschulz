from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

# Funkcja rysująca granice decyzyjne klasyfikatora dla danych 2D zaczerpnięta z:
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/optional-py-scripts/ch03.py

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    # markers = ('o', 'o', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'gold', 'violet', 'chocolate', 'deepskyblue', 'purple')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    # marker=markers[idx],
                    marker='o',
                    label=cl,
                    edgecolor='black')



# Funkcja zamienia obraz RGB na zbiór danych uczących dzieląc obraz na 
# fragmenty o rozmiarze patch_size. 
def img_to_vectors(img, patch_size=(3, 3)):

    img_b = np.pad(img, ((0, patch_size[0]), (0, patch_size[1]), (0,0)), mode='edge')
    
    vectors = []
    for y in range(0, img.shape[0], patch_size[0]):
        for x in range(0, img.shape[1], patch_size[1]):
            patch = img_b[y:y + patch_size[0], x:x + patch_size[1]]
            vectors.append(patch.ravel())
    return np.array(vectors)


# Funkcja odtwarza obraz ze zbioru wektorów uzyskanych z funkcji img_to_vectors().
def vectors_to_image(vectors, img_shape, patch_size):

    img_height, img_width = img_shape[0], img_shape[1]
    restored = np.zeros((img_height + patch_size[0], img_width + patch_size[1], img_shape[2]), dtype=np.int32)

    i=0
    for y in range(0, img_height, patch_size[0]):
        for x in range(0, img_width, patch_size[1]):
            restored[y:y+patch_size[0], x:x+patch_size[1], :] = vectors[i].reshape((patch_size[0], patch_size[1], 3))
            i = i + 1

    return restored[:img_height, :img_width]
