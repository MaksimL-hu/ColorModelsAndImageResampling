import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# Загрузка изображения
# =========================
def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def save_image(img, name):
    Image.fromarray(img.astype(np.uint8)).save(name)

# =========================
# RGB компоненты
# =========================
def split_rgb(img):
    r = np.zeros_like(img)
    g = np.zeros_like(img)
    b = np.zeros_like(img)

    r[:,:,0] = img[:,:,0]
    g[:,:,1] = img[:,:,1]
    b[:,:,2] = img[:,:,2]

    return r,g,b

# =========================
# RGB -> HSI
# =========================
def rgb_to_hsi(img):

    img = img / 255.0

    R,G,B = img[:,:,0], img[:,:,1], img[:,:,2]

    I = (R+G+B)/3

    num = 0.5*((R-G)+(R-B))
    den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + 1e-6

    theta = np.arccos(num/den)

    H = np.where(B<=G, theta, 2*np.pi-theta)
    H = H/(2*np.pi)

    S = 1 - 3*np.minimum(np.minimum(R,G),B)/(R+G+B+1e-6)

    return H,S,I

# =========================
# Инверсия яркости
# =========================
def invert_intensity(img):

    H,S,I = rgb_to_hsi(img)

    I2 = 1 - I

    R,G,B = img[:,:,0]/255, img[:,:,1]/255, img[:,:,2]/255

    k = I2/(I+1e-6)

    R2 = np.clip(R*k,0,1)
    G2 = np.clip(G*k,0,1)
    B2 = np.clip(B*k,0,1)

    result = np.stack((R2,G2,B2),axis=2)*255

    return result.astype(np.uint8)

# =========================
# Интерполяция (nearest)
# =========================
def interpolate(img, M):

    h,w,c = img.shape

    new_h = int(h*M)
    new_w = int(w*M)

    y = (np.arange(new_h)/M).astype(int)
    x = (np.arange(new_w)/M).astype(int)

    return img[y[:,None], x]

# =========================
# Децимация
# =========================
def decimate(img, N):
    return img[::N, ::N]

# =========================
# Передискретизация за 1 проход
# =========================
def resample_one_pass(img, K):

    h,w,c = img.shape

    new_h = int(h*K)
    new_w = int(w*K)

    y = (np.arange(new_h)/K).astype(int)
    x = (np.arange(new_w)/K).astype(int)

    return img[y[:,None], x]

# =========================
# MAIN
# =========================

img = load_image("input.png")

# RGB
r,g,b = split_rgb(img)

# HSI
H,S,I = rgb_to_hsi(img)
I_img = (I*255).astype(np.uint8)

# инверсия
inv = invert_intensity(img)

# передискретизация
M = 3
N = 2
K = M/N

stretch = interpolate(img,M)
compress = decimate(img,N)
two_pass = decimate(stretch,N)
one_pass = resample_one_pass(img,K)

# =========================
# Сохранение
# =========================

save_image(r,"R.png")
save_image(g,"G.png")
save_image(b,"B.png")
save_image(I_img,"Intensity.png")
save_image(inv,"Inverted.png")
save_image(stretch,"Stretch.png")
save_image(compress,"Compress.png")
save_image(two_pass,"Two_pass.png")
save_image(one_pass,"One_pass.png")

# =========================
# Красивый вывод
# =========================

fig, ax = plt.subplots(3,3, figsize=(10,10))

images = [
    (img, "Original"),
    (r, "R"),
    (g, "G"),
    (b, "B"),
    (I_img, "Intensity"),
    (inv, "Intensity inverted"),
    (stretch, "Stretch"),
    (compress, "Compress"),
    (one_pass, "Resample (1 pass)")
]

for a, (image, title) in zip(ax.ravel(), images):

    if title == "Intensity":
        a.imshow(image, cmap="gray")
    else:
        a.imshow(image)

    a.set_title(title, color="black", fontsize=12)
    a.axis("off")

plt.tight_layout()
plt.show()