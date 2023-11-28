import cv2
import numpy as np

def generate_watermark(seed_path, shape):
    # Ubah gambar yang dijadikan seed ke grayscale
    seed_image = cv2.imread(seed_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # Ubah ukuran gambar yang dijadikan seed sehingga mirip dengan gambar yang akan di-watermark
    seed_image = cv2.resize(seed_image, (shape[1], shape[0]))
    
    # Normalisasi ke range nilai 0 dan 1
    seed_image = seed_image / 255.0
    
    # Masukkan gambar yang sebagai seed untuk pseudorandom number generator
    np.random.seed(int(np.sum(seed_image)))
    
    # Buat pola noise dari pseudorandom number generator
    watermark = np.random.rand(*shape).astype(np.float32)
    
    # Ubah pola noise ke pola biner 1 dan -1
    watermark = np.where(watermark < 0.5, -1, 1)
    
    return watermark


def embed_watermark(image, watermark, multiplication_factor):
    # Tambahkan pola noise ke gambar (sesuai rumus di buku acuan)
    watermarked_image = image + multiplication_factor * watermark

    return watermarked_image

def apply_edge_enhance_filter(image):
    # Definisikan kernel (dari buku acuan)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 2

    # Konvolusikan gambar dengan kernel untuk meperjelas edge di gambar tersebut
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    # Normalisasi ke range nilai 0 dan 255
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return filtered_image


def detect_watermark(watermarked_image, watermark):
    # Korelasikan gambar hasil watermark dengan ppola watermark (sesuai buku acuan)
    correlation = np.correlate(watermarked_image.flatten(), watermark.flatten())

    return round(correlation[0], 2)  # Bulatkan


print('\nKuis Watermarking Citra\nII3150 - Sistem Multimedia\n\nNama  : Muhammad Rafi Haidar\nNIM   : 18221134\nKelas : K2\n')

# Minta input dari pengguna untuk scale bobot pengali (k) yang akan dikalikan ke pola watermark saat proses watermarking
multiplication_factor = float(input("Masukkan nilai k: "))


# Ambil gambar yang akan di-watermark dan ubah ke grayscale
image = cv2.imread('./assets/image.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Buat pola untuk watermark
watermark = generate_watermark('./assets/seed.jpg', image.shape)

# Proses watermarking
watermarked_image = embed_watermark(image, watermark, multiplication_factor)

# Tambahkan filter yang akan mengonvolusi gambar yang telah di-watermark untuk memperjelas edge-nya
filtered_image = apply_edge_enhance_filter(watermarked_image)


# Mendeteksi apakah gambar telah di-watermark
# Gambar belum di-watermark jadi nilai korelasinya  kecil
correlation_before_watermarking = detect_watermark(image, watermark)
print('\nKorelasi sebelum proses watermarking:', correlation_before_watermarking)

# Mendeteksi apakah gambar telah di-watermark
# Gambar sudah di-watermark jadi nilai korelasinya lebih besar dari sebelumnya
correlation_after_watermarking = detect_watermark(watermarked_image, watermark)
print('Korelasi setelah proses watermarking:', correlation_after_watermarking)

# Mendeteksi apakah gambar telah di-watermark
# Gambar sudah difilter jadi nilai korelasinya paling besar
correlation_after_enhancing = detect_watermark(filtered_image, watermark)
print('Korelasi setelah proses watermarking dan pemberian filter:', correlation_after_enhancing)


# Simpan hasil di folder results
cv2.imwrite('./results/image_greyscale.jpg', image)                                      # Gambar yang diubah ke grayscale
cv2.imwrite('./results/watermark.jpg', ((watermark + 1) / 2 * 255).astype(np.uint8))     # Pola watermark
cv2.imwrite('./results/watermarked_image.jpg', watermarked_image)                        # Gambar yang di-watermark
cv2.imwrite('./results/watermarked_image_enhanced.jpg', filtered_image)                  # Gambar yang di-watermark dan dikonvolusi