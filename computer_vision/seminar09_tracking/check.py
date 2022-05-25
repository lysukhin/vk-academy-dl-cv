import inspect
import numpy as np


EPS = 1e-3


def almost_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    for va, vb in zip(a.flatten(), b.flatten()):
        if abs(va - vb) > EPS:
            return False
    return True


def is_diag(A):
    diag_A = np.diag(np.diag(A))
    return almost_equal(A, diag_A)


def check_height_16_2(height_16_2, detections):
    if height_16_2 != detections[15][1][3]:
        print("Неверное значение высоты 2-го прямоугольника на 16-м кадре.")
        print("HINT: Возможно, вы используете индексацию начиная с 1 вместо 0.")
        print("HINT: Каждый прямоугольник хранится в формате LTWH.")
        return False
    print("Результат: отлично!")
    return True


def check_ltwh2xysr(fn):
    example = [5.5, 1.2, 4.4, 0.2]
    target = [7.7, 1.3, 2.3, 22]
    result = fn(example)
    try:
        len(result)
    except Exception:
        print("Резyльтат должен быть списком или массивом.")
        return False
    if len(result) != 4:
        print("На выходе должно быть 4 числа")
        return False
    names = "xysr"
    for name, r, t in zip(names, result, target):
        if abs(r - t) > EPS:
            print("Неверно оценён параметр {}".format(name))
            return False
    print("Результат: отлично!")
    return True


def check_xysr2ltwh(fn):
    example = [7.7, 1.3, 2.3, 22]
    target = [5.5, 1.2, 4.4, 0.2]
    result = fn(example)
    try:
        len(result)
    except Exception:
        print("Резyльтат должен быть списком или массивом.")
        return False
    if len(result) != 4:
        print("На выходе должно быть 4 числа")
        return False
    names = "xysr"
    for name, r, t in zip(names, result, target):
        if abs(r - t) > EPS:
            print("Неверно оценён параметр {}".format(name))
            return False
    print("Результат: отлично!")
    return True


def check_get_F(fn):
    F = np.asarray(fn())
    target = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    if F.shape != (7, 7):
        print("Неверный размер выходной матрицы.")
        print("HINT: размер состояния 7")
        return False
    names = ["x", "y", "s", "r", "x'", "y'", "s'"]
    for i, (name, row_F, row_target) in enumerate(zip(names, F, target)):
        if not almost_equal(row_F, row_target):
            print("Неверно предсказывается значение {} (строка {})".format(name, i))
            if name in ["x", "y", "s"]:
                print("NOTE: Новое значение {} зависит только от старого значения {} и скорости.".format(
                    name, name))
            if name == "r":
                print("NOTE: В рамках модель параметр r (соотношение сторон) не меняется.")
            if name in ["x'", "y'", "s'"]:
                print("NOTE: В рамках модели скорость не меняется.")
            return False
    print("Результат: отлично!")
    return True


def check_get_Q(fn):  
    Q = np.asarray(fn())
    parameters = inspect.signature(fn).parameters
    target_diag = np.asarray([parameters[name].default
                              for name in ["pos_std", "pos_std", "scale_std", "aspect_std", "pos_velocity_std",
                                           "pos_velocity_std", "scale_velocity_std"]]) ** 2
    if Q.shape != (7, 7):
        print("Неверный размер выходной матрицы.")
        print("HINT: размер состояния 7 ([x, y, s, r, x', y', s'])")
        return False
    if not is_diag(Q):
        print("Матрица должна быть диагональной, т.к. мы не учитываем зависимости между ошибками разных параметров")
        return False
    Q_diag = np.diag(Q)
    names = ["x", "y", "s", "r", "x'", "y'", "s'"]    
    for i, (name, v_Q, v_target) in enumerate(zip(names, Q_diag, target_diag)):
        if abs(v_Q - v_target) > EPS:
            print("Неверное значение дисперсии для параметра {} (строка {})".format(name, i))
            print("HINT: В параметрах функции даны стандартные отклонения. В матрице ковариации на диагонали должны быть дисперсии")
            return False
    print("Результат: отлично!")
    return True


def check_get_H(fn):
    H = np.asarray(fn())
    target = np.eye(4, 7)
    if H.shape != (4, 7):
        print("Неверный размер матрицы.")
        print("HINT: Число строк должно совпадать с размерностью наблюдений (4, [x, y, s, r])")
        print("HINT: Число столбцов должно совпадать с размерностью состояния (7, [x, y, s, r, x', y', s'])")
        return False
    if not almost_equal(H, target):
        print("Неверные значния матрицы")
        print("HINT: Умножение матрицы H на столбец [x, y, s, r, x', y', s'] должно давать [x, y, s, r]")
        return False
    print("Результат: отлично!")
    return True


def check_get_R(fn):
    R = np.asarray(fn())
    parameters = inspect.signature(fn).parameters
    target_diag = np.asarray([parameters[name].default
                              for name in ["pos_std", "pos_std", "scale_std", "aspect_std"]]) ** 2
    if R.shape != (4, 4):
        print("Неверный размер выходной матрицы.")
        print("HINT: размер наблюдения 4 ([x, y, s, r])")
        return False
    if not is_diag(R):
        print("Матрица должна быть диагональной, т.к. мы не учитываем зависимости между ошибками разных параметров")
        return False
    R_diag = np.diag(R)
    names = ["x", "y", "s", "r"]
    for i, (name, v_R, v_target) in enumerate(zip(names, R_diag, target_diag)):
        if abs(v_R - v_target) > EPS:
            print("Неверное значение дисперсии для параметра {} (строка {})".format(name, i))
            print("HINT: В параметрах функции даны стандартные отклонения. В матрице ковариации на диагонали должны быть дисперсии")
            return False
    print("Результат: отлично!")
    return True


def check_get_P(fn):
    P = np.asarray(fn())
    parameters = inspect.signature(fn).parameters
    target_diag = np.asarray([parameters[name].default
                              for name in ["pos_std", "pos_std", "scale_std", "aspect_std", "pos_velocity_std",
                                           "pos_velocity_std", "scale_velocity_std"]]) ** 2
    if P.shape != (7, 7):
        print("Неверный размер выходной матрицы.")
        print("HINT: размер состояния 7 ([x, y, s, r, x', y', s'])")
        return False
    if not is_diag(P):
        print("Матрица должна быть диагональной, т.к. мы не учитываем зависимости между ошибками разных параметров")
        return False
    P_diag = np.diag(P)
    names = ["x", "y", "s", "r", "x'", "y'", "s'"]    
    for i, (name, v_P, v_target) in enumerate(zip(names, P_diag, target_diag)):
        if abs(v_P - v_target) > EPS:
            print("Неверное значение дисперсии для параметра {} (строка {})".format(name, i))
            print("HINT: В параметрах функции даны стандартные отклонения. В матрице ковариации на диагонали должны быть дисперсии")
            return False
    print("Результат: отлично!")
    return True

def check_batch_iou(fn):
    bboxes1 = np.array([[0, 0, 4, 4],
                        [0, 0, 4, 6]])
    bboxes2 = np.array([[0, 0, 4, 4],
                        [-2, -2, 4, 4],
                        [-4, -4, 4, 4],
                        [-6, -6, 4, 4]])
    iou_gt = np.array([[1    , 1 / 7, 0, 0],
                       [2 / 3, 1 / 9, 0, 0]])
    result = np.asarray(fn(bboxes1, bboxes2))
    if result.shape != (2, 4):
        print("Неверный размер выходной матрицы")
        return False
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            if abs(result[i, j] - iou_gt[i, j]) > EPS:
                print("Неверный ответ")
                print("Prediction:", bbox1)
                print("Detection:", bbox2)
                print("Ответ:", result[i, j])
                print("Правильный ответ:", iou_gt[i, j])
                return False
    print("Результат: отлично!")
    return True


def check_match_bboxes(fn):
    np.random.seed(31415)
    # Check shapes and values.
    for size1 in range(5):
        for size2 in range(5):
            iou = 0.5 * np.random.random((size1, size2)) + 0.5  # > 0.
            matches = fn(iou)
            if len(matches) != size1:
                print("Неверный размер выходного массива.")
                print("HINT: Ожидается одномерный массив длины, равной числу строк матрицы iou")
                return False
            no_none = [m for m in matches if m is not None]
            if no_none and np.max(np.bincount(no_none)) > 1:
                print("Нельзя связывать предсказание с несколькими детектами")
                return False
            if len(no_none) != min(size1, size2):
                print("Недостаточное число соответствий")
                return False
            if no_none and (max(no_none) >= size2):
                print("Очень большой выходной индекс")
                return False
    
    # Check simple cases.
    iou = np.zeros((10, 1))
    matches = fn(iou)
    if [m for m in matches if m is not None]:
        print("Неверный ответ для матрицы:")
        print(iou)
        print("HINT: Нельзя связывать при нулевом IoU")
        return False
    
    iou = 0.9 * np.diag(np.random.random(10)) + 0.1
    matches = fn(iou)
    if (None in matches) or (not almost_equal(matches, np.arange(10))):
        print("Неверный ответ для матрицы:")
        print(iou)
        return False
    
    print("Результат: отлично!")
    return True


def check_batch_cosine_distance(fn):
    embeddings1 = np.random.random((5, 12))
    embeddings2 = np.random.random((8, 12))
    result = np.asarray(fn(embeddings1, embeddings2))
    if result.shape != (5, 8):
        print("Неверный размер выходной матрицы")
        return False
    result_scaled = np.asarray(fn(embeddings1 * 18, embeddings2 / 10))
    if not almost_equal(result, result_scaled):
        print("Результат не должен зависеть от длины вектора")
        return False
    embeddings1 = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float)
    embeddings2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float)
    gt = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float)
    result = fn(embeddings1, embeddings2)
    if not almost_equal(result, gt):
        print("Неверный ответ")
        return False
    
    embeddings1 = np.zeros((2, 128))
    embeddings1[:, 64] = 1
    embeddings2 = np.ones((4, 128))
    gt = np.ones((2, 4)) / np.sqrt(128)
    result = fn(embeddings1, embeddings2)
    if not almost_equal(result, gt):
        print("Неверный ответ")
        return False
    
    print("Результат: отлично!")
    return True


def check_batch_similarity(fn):
    alpha = np.random.random()
    bboxes1 = np.array([[0, 0, 4, 4],
                        [0, 0, 4, 6]])
    bboxes2 = np.array([[0, 0, 4, 4],
                        [-2, -2, 4, 4],
                        [-4, -4, 4, 4],
                        [-6, -6, 4, 4]])
    iou = np.array([[1    , 1 / 7, 0, 0],
                    [2 / 3, 1 / 9, 0, 0]])
    embeddings1 = np.zeros((2, 128))
    embeddings1[:, 64] = 1
    embeddings2 = np.ones((4, 128))
    cos = np.ones((2, 4)) / np.sqrt(128)
    result = np.asarray(fn(bboxes1, embeddings1, bboxes2, embeddings2, alpha=alpha))
    if result.shape != (2, 4):
        print("Неверный размер выходной матрицы")
        return False
    
    gt = alpha * iou + (1 - alpha) * cos
    if not almost_equal(result, gt):
        print("Неверный результат")
        print("HINT: Проверьте, что правильно используется параметр alpha")
        return False
    print("Результат: отлично!")
    return True