subroutine calc_kernel(array, NX, NY, px, py, P, arr)

real(8), dimension(0:NX-1, 0:NY-1), intent(in) :: array
integer, intent(in) :: NX
integer, intent(in) :: NY
integer, intent(in) :: px
integer, intent(in) :: py
integer, intent(in) :: P

real(8), dimension(0:NX-1, 0:NY-1), intent(out) :: arr

integer :: i, j, k, point
real(8) :: a

arr = array

! Внутренние точки
do i = 1, NX-1-1

    do j = 1, NY-1-1
        arr(i,j) = arr(i-1, j) + arr(i+1, j) 
        arr(i,j) = arr(i,j) + arr(i, j-1) + arr(i, j+1) 
        arr(i,j) = arr(i,j) / 4.0
    enddo

enddo

! Потолок
arr(:, NY-1) = 2.0 * arr(:, NY-1-1) - arr(:, NY-1-2)

! Бок
arr(NX-1, :) = 2.0 * arr(NX-1-1, :) - arr(NX-1-2, :)

! Пластина
a = 0
k = py

do point = 0, P-1
    a = a + arr(k, px)
    k = k + 1
enddo

a = a / real(P)

k = py
do point = 0, P-1
    k = k + 1
    arr(k, px) = a 
enddo

end subroutine