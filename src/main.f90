!gfortran -fdefault-real-8 json.f90 bfgs.f90 main.f90 -o bfgs.out
!g95 -r8 bfgs.f90 main.f90 -o bfgs.out

program main
    use optix
    implicit none
    character(100) :: fn

    write(*,*) '-----------------------------------------------'
    write(*,*) '|                  Optix 1.1                  |'
    write(*,*) '|                                             |'
    write(*,*) '|           (c) Doug Hunsaker, 2015           |'
    write(*,*) '|                                             |'
    write(*,*) '|          This software comes with           |'
    write(*,*) '| ABSOLUTELY NO WARRANTY EXPRESSED OR IMPLIED |'
    write(*,*) '|                                             |'
    write(*,*) '|        Submit bug reports online at:        |'
    write(*,*) '|            www.doughunsaker.com             |'
    write(*,*) '-----------------------------------------------'

    !Initialize
    pi = 3.14159265358979323
    iter = 0

    call get_command_argument(1,inputfile)
    call opt_read_json()
    
    fn = 'optix_start.json'; call write_optix_file(fn)
    call opt_run()
    fn = 'optix_end.json'; call write_optix_file(fn)

    call opt_deallocate()
    write(*,*) 'Optimization Complete.'
end program main



