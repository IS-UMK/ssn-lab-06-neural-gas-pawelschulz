import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from utils import img_to_vectors, vectors_to_image
from neural_gas import NeuralGas
import os

def mse(img1, img2):
    return np.sum((img1.astype(np.float32) - img2.astype(np.float32)) ** 2) / img1.size

# Tworzenie folderu do zapisywania wyników (jeśli nie istnieje)
output_dir = 'wyniki'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Wczytanie obrazu Lenna
img = imread('dane/Lenna.png')  # Podaj poprawną ścieżkę do pliku
plt.imshow(img)
plt.title("Oryginalny obraz Lenna")
plt.axis('off')
# Zapisanie oryginalnego obrazu
plt.savefig(os.path.join(output_dir, 'original_image.png'), bbox_inches='tight')
plt.show()

# Parametry
patch_size = (5, 5)
n_prototypes = 64
n_epochs = 10

# Przygotowanie danych
X = img_to_vectors(img, patch_size)
X = X / 255  # Normalizacja

# Uczenie modelu
model = NeuralGas(n_prototypes=n_prototypes, eta0=0.3, lambda0=5.0, 
                  eta_min=0.05, lambda_min=0.5, n_epochs=n_epochs)
model.fit(X)

# Rekonstrukcja obrazu
pred = np.array([model.find_nearest_prototype(x) for x in X])
restored_img = vectors_to_image(model.prototypes[pred] * 255, img.shape, patch_size)

# Wyświetlenie zrekonstruowanego obrazu
plt.imshow(restored_img.astype(np.uint8))
plt.title(f"Zrekonstruowany obraz (k={n_prototypes})")
plt.axis('off')
# Zapisanie zrekonstruowanego obrazu
plt.savefig(os.path.join(output_dir, f'restored_image_k{n_prototypes}.png'), bbox_inches='tight')
plt.show()

# Błąd rekonstrukcji
error = mse(img, restored_img)
print(f"Błąd rekonstrukcji (MSE): {error:.2f}")

# Wykres błędu podczas uczenia
plt.plot(model.errors)
plt.xlabel("Epoka")
plt.ylabel("Błąd (MSE)")
plt.title("Spadek błędu podczas uczenia gazu neuronowego")
plt.grid(True)
# Zapisanie wykresu błędu
plt.savefig(os.path.join(output_dir, f'error_plot_k{n_prototypes}.png'), bbox_inches='tight')
plt.show()

# Zapisanie wartości MSE do pliku tekstowego
with open(os.path.join(output_dir, f'error_values_k{n_prototypes}.txt'), 'w') as f:
    f.write(f"Błąd rekonstrukcji (MSE): {error:.2f}\n")
    f.write("Wartości błędu w trakcie uczenia:\n")
    for epoch, err in enumerate(model.errors):
        f.write(f"Epoka {epoch + 1}: {err:.6f}\n")
