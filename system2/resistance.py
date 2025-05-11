import numpy as np

def ellipse(a, b):
    """ellipse resistance

    Args:
        a (float): major axis
        b (float): minor axis
    """
    if a < b:
        tmp = a
        a = b
        b = tmp
    
    gamma = a / b
    alpha = 4 * np.pi * (gamma + (1 / gamma))
    area = np.pi * a * b
    return alpha, area

def rectangular(w, h, n):
    """rectangular resistance

    Args:
        w (float): width
        h (float): height
    """
    f = lambda n: (1 / (n**5)) * np.tanh(n * np.pi * w / (2 * h))
    g = lambda n: sum([f(i) for i in range(1, n, 2)])
    
    alpha = 12 * (w / h) * (1 - (h / w) * ((192 / np.pi**5) * g(n)))
    area = w * h
    return alpha, area

def triangular(a, b, c):
    """triangular resistance

    Args:
        a (float): side a
        b (float): side b
        c (float): side c
    """
    C = (8 * (a + b + c)**2) / ((0.5 * (a**2 + b**2 + c**2)**2 - (a**4 + b**4 + c**4))**0.5)
    alpha = (25 / 17) * C + (40 * np.sqrt(3) / 17)
    s = 0.5 * (a + b + c)
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return alpha, area

def harmonical_perturbed_circle(epsilon, k):
    """harmonical perturbed circle resistance

    Args:
        epsilon (float): perturbation parameter
        k (int): order of the harmonic perturbation (k > 2)
    """
    C = 4 * np.pi + 2 * np.pi * (k ** 2 - 1) * epsilon ** 2
    alpha = (8 / (1 + k)) * C - 8 * ((3 - k) / (1 + k)) * np.pi
    area = np.pi
    return alpha, area