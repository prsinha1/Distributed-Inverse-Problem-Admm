import numpy as np
import matplotlib.pyplot as plt
import odl

def ConstIm(N):
    mRec = N // 2
    rCir1 = N // 2
    rCir2 = N // 8
    rCir3 = N // 16

    Im = np.zeros((N, N))

    # Fixed rectangle placement
    rec1s = N // 6
    rec1e = min(N, rec1s + mRec - 1)
    rec2s = N // 5
    rec2e = min(N, rec2s + mRec - 1)
    Im[rec1s:N, rec2s:N] = 200

    # Large circle, fixed center
    Im_tmp = np.zeros((N, N))
    ctr1 = (N // 3, N // 3)  # fixed center
    I1 = np.arange(max(ctr1[0] - rCir1, 0), min(ctr1[0] + rCir1, N))
    I2 = np.arange(max(ctr1[1] - rCir1, 0), min(ctr1[1] + rCir1, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr1[0]) ** 2 + (Y - ctr1[1]) ** 2) <= rCir1 ** 2
    cir = cir.astype(float) * 80
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.where(Im_tmp == 0, Im, Im_tmp)

    # Medium circle, fixed center
    Im_tmp = np.zeros((N, N))
    ctr2 = (3 * N // 5, 3 * N // 5)  # fixed center
    I1 = np.arange(max(ctr2[0] - rCir2, 0), min(ctr2[0] + rCir2, N))
    I2 = np.arange(max(ctr2[1] - rCir2, 0), min(ctr2[1] + rCir2, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr2[0]) ** 2 + (Y - ctr2[1]) ** 2) <= rCir2 ** 2
    cir = cir.astype(float) * 300
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.maximum(Im, Im_tmp)

    # Small circle near top-right corner
    Im_tmp = np.zeros((N, N))
    ctr3 = (N // 10, N - N // 6)  # fixed center
    I1 = np.arange(max(ctr3[0] - rCir3, 0), min(ctr3[0] + rCir3, N))
    I2 = np.arange(max(ctr3[1] - rCir3, 0), min(ctr3[1] + rCir3, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr3[0]) ** 2 + (Y - ctr3[1]) ** 2) <= rCir3 ** 2
    cir = cir.astype(float) * 400
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.maximum(Im, Im_tmp)

    # Small circle near bottom-left corner
    Im_tmp = np.zeros((N, N))
    ctr4 = (N - N // 6, N // 10)  # fixed center
    I1 = np.arange(max(ctr4[0] - rCir3, 0), min(ctr4[0] + rCir3, N))
    I2 = np.arange(max(ctr4[1] - rCir3, 0), min(ctr4[1] + rCir3, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr4[0]) ** 2 + (Y - ctr4[1]) ** 2) <= rCir3 ** 2
    cir = cir.astype(float) * 400
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.maximum(Im, Im_tmp)

    return Im


def randIm(N):
    mRec = N // 2
    rCir1 = N // 2
    rCir2 = N // 8
    rCir3 = N // 16

    Im = np.zeros((N, N))

    ofsRec = np.random.randint(N // 8, N // 4 + N // 8, size=2)
    rec1s = ofsRec[0]
    rec1e = min(N, rec1s + mRec - 1)
    rec2s = ofsRec[1]
    rec2e = min(N, rec2s + mRec - 1)
    Im[rec1s:N, rec2s:N] = 200

    Im_tmp = np.zeros((N, N))
    ctr1 = np.random.randint(N // 4, N // 2, size=2)
    I1 = np.arange(max(ctr1[0] - rCir1, 0), min(ctr1[0] + rCir1, N))
    I2 = np.arange(max(ctr1[1] - rCir1, 0), min(ctr1[1] + rCir1, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr1[0]) ** 2 + (Y - ctr1[1]) ** 2) <= rCir1 ** 2
    cir = cir.astype(float) * 80
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.where(Im_tmp == 0, Im, Im_tmp)

    Im_tmp = np.zeros((N, N))
    ctr2 = np.random.randint(N // 2, 3 * N // 4, size=2)
    I1 = np.arange(max(ctr2[0] - rCir2, 0), min(ctr2[0] + rCir2, N))
    I2 = np.arange(max(ctr2[1] - rCir2, 0), min(ctr2[1] + rCir2, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr2[0]) ** 2 + (Y - ctr2[1]) ** 2) <= rCir2 ** 2
    cir = cir.astype(float) * 300
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.maximum(Im, Im_tmp)

    Im_tmp = np.zeros((N, N))
    ctr3 = np.random.randint(0, N // 4, size=2) + np.array([0, N - N // 4])
    I1 = np.arange(max(ctr3[0] - rCir3, 0), min(ctr3[0] + rCir3, N))
    I2 = np.arange(max(ctr3[1] - rCir3, 0), min(ctr3[1] + rCir3, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr3[0]) ** 2 + (Y - ctr3[1]) ** 2) <= rCir3 ** 2
    cir = cir.astype(float) * 400
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.maximum(Im, Im_tmp)

    Im_tmp = np.zeros((N, N))
    ctr3 = np.random.randint(0, N // 4, size=2) + np.array([N - N // 4, 0])
    I1 = np.arange(max(ctr3[0] - rCir3, 0), min(ctr3[0] + rCir3, N))
    I2 = np.arange(max(ctr3[1] - rCir3, 0), min(ctr3[1] + rCir3, N))
    X, Y = np.meshgrid(I1, I2)
    cir = ((X - ctr3[0]) ** 2 + (Y - ctr3[1]) ** 2) <= rCir3 ** 2
    cir = cir.astype(float) * 400
    Im_tmp[np.ix_(I2, I1)] = cir
    Im = np.maximum(Im, Im_tmp)

    return Im

def generate_sinogram(image, angles, plot_sparsity=True):
    N = image.shape[0]
    space = odl.uniform_discr([-1, -1], [1, 1], [N, N], dtype='float32')
    phantom = space.element(image)

    angle_partition = odl.uniform_partition(0, np.pi, len(angles))
    detector_partition = odl.uniform_partition(-1.0, 1.0, N)

    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='skimage')
    sinogram = ray_trafo(phantom)
    noisy_sinogram = sinogram + odl.phantom.white_noise(ray_trafo.range) * 0.0000000000005

    # === Construct forward matrix A ===
    domain_size = ray_trafo.domain.size
    range_size = ray_trafo.range.size
    A = np.zeros((range_size, domain_size))
    for j in range(domain_size):
        e_j = ray_trafo.domain.zero()
        idx = np.unravel_index(j, ray_trafo.domain.shape)
        e_j[idx] = 1.0
        A[:, j] = ray_trafo(e_j).asarray().flatten()

    return noisy_sinogram, ray_trafo, geometry, space, A


    # Sparsity pattern and rank (if requested)
    if plot_sparsity:
        A = ray_trafo.asarray()
        rank = np.linalg.matrix_rank(A)
        print(f"RayTransform matrix rank: {rank}")

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for i in range(5):
            if i < len(angles):
                axes[i].imshow(A[i::len(angles)], cmap='gray', aspect='auto')
                axes[i].set_title(f"Phantom {i+1}")
                axes[i].axis('off')
        plt.suptitle("Sparsity Patterns of RayTransform Matrices")
        plt.tight_layout()
        plt.show()

   
