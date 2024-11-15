from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    #raise NotImplementedError
    x = np.load(filename)
    mean_image = np.mean(x, axis=0)
    center_data = x - mean_image
    return center_data
#     Your implementation goes here!
# n (number of image) =13233
# m (number of features for each sample image) =4096


def get_covariance(dataset):
    # Your implementation goes here!
    #raise NotImplementedError
    x = dataset.shape[0]
    covariance_matrix = (1/(x-1))*(dataset.T @ dataset)
    return covariance_matrix

def get_eig(S, k):
    #raise NotImplementedError
    n = S.shape[0]
    eigenvalues, eigenvectors = eigh(S, subset_by_index = [n-k, n-1])
    #idx = np.argsort(eigenvalues)[::-1][:k]
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    Lambda = np.diag(eigenvalues)
    U = eigenvectors
    #top_eigenvalues = np.diag(eigenvalues[idx])
    #top_eigenvectors = eigenvectors[:, idx]
    return Lambda, U


# +
# def get_eig_prop(S, prop):
#     #raise NotImplementedError
    
#     eigenvalues, eigenvectors = eigh(S)
#     eigenvalues = eigenvalues[::-1]
#     eigenvectors = eigenvectors[:, ::-1]
#     sum_var = np.sum(eigenvalues)
#     cum_var = np.cumsum(eigenvalues) / sum_var
#     num_components = np.searchsorted(cum_var, prop) + 1
#     top_eigenvalues = np.diag(eigenvalues[:num_components])
#     top_eigenvectors = eigenvectors[:, :num_components]
#     return top_eigenvalues, top_eigenvectors


# +
import numpy as np
from scipy.linalg import eigh

def get_eig_prop(S, prop):
    # Step 1: Compute all eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(S)
    
    # Step 2: Sort eigenvalues in descending order (and sort eigenvectors accordingly)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices for sorting in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 3: Compute total variance (sum of all eigenvalues)
    total_variance = np.sum(eigenvalues)
    
    # Step 4: Calculate the proportion of variance explained by each eigenvalue
    explained_variance_ratios = eigenvalues / total_variance
    
    # Step 5: Cumulative sum of explained variance to determine how many eigenvalues to keep
    cumulative_variance = np.cumsum(explained_variance_ratios)
    
    # Step 6: Select eigenvalues and eigenvectors that explain more than the given proportion
    k = np.searchsorted(cumulative_variance, prop) + 1  # Find the smallest index where cum_var exceeds prop
    
    # Step 7: Return the top-k eigenvalues as a diagonal matrix and the corresponding eigenvectors
    Lambda = np.diag(eigenvalues[:k])
    U = eigenvectors[:, :k]
    
    return Lambda, U



# -

def project_image(image, U):
    #raise NotImplementedError
    
    projection = U.T @ image
    projected_image = U @ projection
    return projected_image

def display_image(orig, proj):
    #raise NotImplementedError
    
    orig_img = orig.reshape((64, 64))
    proj_img = proj.reshape((64, 64))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(proj_img, cmap='gray')
    axs[1].set_title('Projected Image')
    plt.show()


def perturb_image(image, U, sigma):
    #raise NotImplementedError
    
    projection = U.T @ image
    noise = np.random.normal(0, sigma, size=projection.shape)
    perturbed_projection = projection + noise
    perturbed_image = U @ perturbed_projection
    return perturbed_image


def combine_image(image1, image2, U, lam):
    #raise NotImplementedError
    
    projection1 = U.T @ image1
    projection2 = U.T @ image2
    combined_projection = lam * projection1 + (1 - lam) * projection2
    combined_image = U @ combined_projection
    return combined_image

# +
#x = load_and_center_dataset('face_dataset.npy')

#print (len(x), len(x[0]), np.average(x))

# +
#LLM : chatgpt
#Questions : what function in python gives you the mean?
# what does np.transpose() do?
#how to sort through eigenvalues? 
# 
# -


