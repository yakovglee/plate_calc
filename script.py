from calculation import *

def main():
    PATH = 'data/data.csv'
    NX, NY, X, Y, alpha, eps = prepare_data('data.csv')
    NX, NY = int(NX), int(NY)
    xx, yy, hx, hy, px, py, P = prepare_field(X, Y, NX, NY)

    z = np.zeros(xx.shape, dtype='f8', order='F')
    z = calculate_init_cond(z, xx, yy, alpha)

    psi_prev = copy.deepcopy(z)

    psi_now, iter_now = calculate_psi(psi_prev, NX, NY, px, py, P, eps=eps)

    cp_theory = calculate_cp_theory(theta=alpha)

    cp_chisl = calculate_cp(psi_now, hx, px, py, P)

    save_data(psi_now, xx, yy, cp_theory, cp_chisl,
              fname_psi='psi_{}'.format(eps),
              fname_cp_th='cp_th_{}'.format(eps),
              fname_cp_ch='cp_ch_{}'.format(eps)
              )


if __name__ == '__main__':
    main()