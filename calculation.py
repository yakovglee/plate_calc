from kernel import calc_kernel
import numpy as np
import pandas as pd
import copy
import sympy as sp
from sympy import re, im, I, E, symbols
import matplotlib.pyplot as plt


def prepare_data(PATH, sep=';'):
    """
    Считывает начальные значения

    Возвращает:
        NX, NY - количество шагов разбиения по оси OX, OY
        X, Y - расчетная область
        alpha - угол атаки в град.
        eps - погрешность
    """

    return np.genfromtxt(PATH, delimiter=sep, skip_header=1)


def prepare_field(X, Y, NX, NY):
    """
    Считывает начальные значения

    Принимает:
        NX, NY - количество шагов разбиения по оси OX, OY
        X, Y - расчетная область

    Возвращает:
        xx, yy - координаты расчетной области
        hx, hy - шаг сетки
        px, py, P - параметры пластины
    """

    px, py = NX // 2, NY // 3
    P = NY - 2 * py

    x = np.linspace(0.0, X, NX)
    y = np.linspace(0.0, Y, NY)

    xx, yy = np.meshgrid(x, y)

    hx = xx[0, 1] - xx[0, 0]
    hy = yy[1, 0] - yy[0, 0]

    return xx, yy, hx, hy, px, py, P


def psi(x, y, alpha=0.0):
    """
    Функция линии тока
    """
    alpha = np.deg2rad(alpha)
    return np.cos(alpha) * y - np.sin(alpha) * x


def calculate_init_cond(array, xx, yy, alpha):
    """
    Вычисляет первое приближение

    Принимает:
        - field:array - вычислительную область
        - psi:function - функция, описывающая начальные условия
        - x, y:array - векторы координат
    Возвращает:
        - :array - вычислительную область с граничными условиями
    """
    arr = copy.deepcopy(array)
    arr = psi(xx, yy, alpha)
    return arr


def calculate_psi(psi_prev, NX, NY, px, py, P, ITER=100000, eps=0.001):
    """
    Основной алгоритм для вычисления пси

    Параметры:
        psi_prev - массив начальных приближений размерности (NX,NY)
        NX, NY - количество шагов разбиения по оси OX, OY
        px, py - координаты точек пластины
        P - количество точек
        ITER - количество итераций
        eps - погрешность

    Возвращает:
        psi - значения psi
        iter_now - итерация, на которой заверщился расчет
    """

    iter_now = 0

    for iter in range(1, ITER):
        psi_now = calc_kernel(psi_prev, px=px, py=py, p=P, nx=NX, ny=NY)
        delta = np.max(np.abs(psi_now - psi_prev))

        psi_prev = copy.deepcopy(psi_now)

        if delta < eps:
            iter_now = iter
            break

    return psi_now, iter_now


def calculate_cp_theory(theta):
    """
    Считает теоретическое распределение давления при вертикальном положении пластины (y=const)

    Принимает:
        theta - угол атаки в градусах

    Возвращает:
        CP_ - распределение давления, посчитанное в 150 точках
    """
    x, y = symbols('x y', real=True)
    alpha = symbols('alpha', real=True)
    a = symbols('a', real=True)
    z = x

    # комплексная скорость
    v = sp.cos(alpha) - I * sp.sin(alpha) * z / sp.sqrt(sp.Pow(z, 2) - sp.Pow(a, 2))

    ugol = np.deg2rad(90 - theta)
    px = np.linspace(-.49, .49, 150)
    L = 0.5

    CP_ = []

    for i in px:
        s = re(v).subs({alpha: ugol, x: i, a: L}).evalf()
        CP_.append(1.0 - s * s)

    return CP_


def calculate_cp(psi, h, px, py, P):
    """
    Считает численное распределение давления

    Принимает:
        psi - массив со значениями функции тока
        h - шаг сетки
        px, py, P - координаты расположения пластины, количество точек на пластине

    Возвращает:
        CP - распределение давления, посчитанное в P точках
    """
    CP = []
    k = py
    for i in range(P):
        vel = (psi[k, px - 1] - psi[k, px + 1]) / 2 / h
        k += 1

        CP.append(1.0 - vel * vel)

    return CP


def save_data(psi, xx, yy, cp_theory, cp_chisl,
              fname_psi='psi',
              fname_cp_th='cp_th',
              fname_cp_ch='cp_ch'):

    data_psi = pd.DataFrame(psi[::-1],
                            columns=xx[0, :],
                            index=yy[:, 0][::-1])

    np.savetxt(fname_cp_th + ".csv", cp_theory, delimiter=";")
    np.savetxt(fname_cp_ch + ".csv", cp_chisl, delimiter=";")
    data_psi.to_csv(fname_psi + '.csv', sep=';')


def draw_and_save_cp(cp_th, cp_ch, fname):
    ymin = -12
    x_ch = np.linspace(0, 1, len(cp_ch))
    x_th = np.linspace(0, 1, len(cp_th))
    x_t = np.arange(0, 1.1, 0.1)

    plt.figure(figsize=(10, 10))
    plt.plot(x_th, cp_th, label='Теория', color='b')
    plt.plot(x_ch, cp_ch, label=r'МКР $\epsilon=0.001$', color='r', ls='-.')
    plt.axvline(0.5 * np.cos(np.deg2rad(15)), ymax=0.915, ls='--', alpha=0.5)

    plt.annotate("Кр. точка 0.48", xy=(0.5 * np.cos(np.deg2rad(15)), -14), xytext=(0.55, -13),
                 arrowprops=dict(arrowstyle="->"), fontsize=13)

    plt.yticks(np.arange(ymin, 2, 1))
    plt.xticks(x_t)

    plt.ylim(-12)
    plt.title(r'$C_p$ при $\alpha$ = 15$^\circ$', fontsize=14)
    plt.xlabel(r'Y', fontsize=14)
    plt.ylabel(r'$C_p$', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid();
    plt.savefig(fname, dpi=200, bbox_inches='tight')


def draw_and_save_psi(psi, px, py, P, fname):
    fig, ax = plt.subplots(figsize=(10, 10))
    contourf_ = ax.contour(psi, [i * 0.2 for i in range(-50, 50)], colors='k', alpha=0.5, linestyles='solid')
    ax.plot([px, px], [py, py + P], lw=2.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(r'Линии тока при $\alpha$ = {}$^\circ$'.format(15))
    fig.savefig(fname, dpi=200, bbox_inches='tight')