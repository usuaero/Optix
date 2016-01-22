module optix
    use myjson_m
    implicit none

    type string_t
        character(100) :: name
    end type string_t


    character(10) :: platform
    character(100) :: inputfile
    character(100) :: varfilename
    character(100) :: fitnessfilename
    character(10) :: diff_scheme ! 'central' or 'forward'

    type(json_file) :: json    !the JSON structure read from the file:

    integer :: nvars
    integer :: nconstraints
    integer :: iter
    integer :: nsearch
    integer :: nruncommands
    integer :: iverbose

    real :: default_alpha
    real :: diff_delta
    real :: fitness_curr
    real :: stop_delta
    real :: alpha

    integer,allocatable :: opton(:), constrainttype(:)
    real,allocatable :: grad(:)
    real,allocatable :: vars(:), constraintvalues(:), penalty(:), penalty_factor(:)
    real,allocatable :: s(:)
    type(string_t),allocatable,dimension(:) :: varnames, constraintnames
    type(string_t),allocatable,dimension(:) :: runcommands
    
    real :: pi
    
contains

!---------------------------------------------------------------------------------------------------------------------------------

subroutine opt_allocate()
    allocate(opton(nvars))
    allocate(grad(nvars))
    allocate(vars(nvars))
    allocate(s(nvars))
    allocate(varnames(nvars))
    opton = 0; vars = 0.0; grad = 0.0; s = 0.0
    
    allocate(constraintnames(nconstraints))
    allocate(constraintvalues(nconstraints))
    allocate(constrainttype(nconstraints))
    allocate(penalty(nconstraints))
    allocate(penalty_factor(nconstraints))
end subroutine opt_allocate

!---------------------------------------------------------------------------------------------------------------------------------

subroutine opt_deallocate()
    deallocate(opton)
    deallocate(grad)
    deallocate(vars)
    deallocate(s)
end subroutine opt_deallocate

!---------------------------------------------------------------------------------------------------------------------------------

subroutine opt_read_json()
    implicit none
    type(json_value),pointer :: j_this, j_var, c_this, c_var
    character(len=:),allocatable :: cval
    character(10) :: istr
    integer :: loc,i
    
    if(iverbose.eq.1) write(*,*) 'Reading input file: ',inputfile
    call json%load_file(filename = inputfile); call json_check()
    loc = index(inputfile,'.json')
    inputfile = inputfile(1:loc-1) !deletes the .json file extension
    
!    call t%json%print_file()
    call json%get('settings.default_alpha',      default_alpha);     call json_check()
    call json%get('settings.diff_delta',            diff_delta);     call json_check()
    call json%get('settings.diff_scheme',                 cval);     call json_check(); diff_scheme = trim(cval)
    call json%get('settings.stop_delta',            stop_delta);     call json_check()
    call json%get('settings.n_search',                 nsearch);     call json_check()
    iverbose = 0
    call json%get('settings.verbose',                 iverbose);
    call json_clear_exceptions()
    write(*,*) 'verbose = ',iverbose


    ! Read run commands
    nruncommands = 0
    call json%get('run', j_this); call json_check();
    nruncommands = json_value_count(j_this)
    write(*,*) 'Commands to run for each case : '
    allocate(runcommands(nruncommands))
    do i=1,nruncommands
        call integer_to_string(i,istr)
        call json%get('run('//trim(istr)//')', cval);     call json_check(); runcommands(i)%name = trim(cval)
        write(*,*) runcommands(i)%name
    end do
    write(*,*)
    
    !Read Interface
    call json%get('interface.variables',          cval);     call json_check(); varfilename = trim(cval)
    call json%get('interface.fitness',            cval);     call json_check(); fitnessfilename = trim(cval)
    write(*,*) 'Writing variables to : ',trim(varfilename)
    write(*,*) 'Reading fitness from : ',trim(fitnessfilename)

    ! Get variable and constraint vector lengths
    call json%get('variables', j_this); call json_check();
    nvars = json_value_count(j_this)
    call json%get('constraints', c_this); 
    if(json_failed()) then
        nconstraints = 0
    else
        nconstraints = json_value_count(c_this)
    end if
    call json_clear_exceptions()
    call opt_allocate()

    ! Read variables
    opton(:) = 0
    do i=1,nvars
        call json_value_get(j_this,i,j_var)
        varnames(i)%name = trim(j_var%name)
        call json%get('variables.'//trim(j_var%name)//'.init', vars(i));  call json_check();
        call json%get('variables.'//trim(j_var%name)//'.opt',  cval);     call json_check();
        if(trim(cval).eq."on") opton(i) = 1
    end do
    write(*,*)

    ! Read constraints
    do i=1,nconstraints
        call json_value_get(c_this,i,c_var)
        constraintnames(i)%name = trim(c_var%name)
        call json%get('constraints.'//trim(c_var%name)//'.value', constraintvalues(i));  call json_check();
        call json%get('constraints.'//trim(c_var%name)//'.type',  cval);     call json_check();
        if(trim(cval).eq."=") constrainttype(i) = 1
        if(trim(cval).eq."<") constrainttype(i) = 2
        if(trim(cval).eq.">") constrainttype(i) = 3
        penalty(i) = json_optional_real(c_var,'penalty',1.0);
        penalty_factor(i) = json_optional_real(c_var,'factor',2.0);
    end do
    write(*,*)

end subroutine opt_read_json

!-----------------------------------------------------------------------------------------------------------
subroutine opt_run()
    integer :: i_iter,o_iter,i,ierror
    real :: vars_orig(nvars),vars_old(nvars),grad_old(nvars)
    real :: dx(nvars,1),NG(nvars,1),N(nvars,nvars),gamma(nvars,1)
    real :: mag_dx,denom
    character(100) :: savefile
    character(LEN=50)::fn,constraint_type
    110 format (1X, I10, 1000ES22.13)

    fn = 'optimization.txt'
    open(unit = 1001, File = fn, action = "write", iostat = ierror)
    write(1001,*) 'iter o_it i_it   Fitness               alpha                 mag(dx)               Vars'
    close(1001)
    open(unit = 1001, File = 'gradient.txt', action = "write", iostat = ierror)
    write(1001,*) 'iter o_it i_it   Fitness               alpha                 mag(dx)               Vars'
    close(1001)


    write(*,*) '---------- Variables ----------'
    do i=1,nvars
        write(*,*) trim(varnames(i)%name),vars(i)
    end do
    write(*,*)

    write(*,*) '---------- Constraints ----------'
    do i=1,nconstraints
        select case (constrainttype(i))
            case (1)
                constraint_type = '='
            case (2)
                constraint_type = '<'
            case (3)
                constraint_type = '>'
            case default
                constraint_type = 'undefined'
        end select
        write(*,*) trim(constraintnames(i)%name),' ',trim(constraint_type),' ',constraintvalues
    end do
    write(*,*)
    write(*,*) '---------- Settings ----------'
    write(*,*) '      default alpha : ',default_alpha
    write(*,*) 'differenceing delta : ',diff_delta
    write(*,*) 'differencing scheme : ',diff_scheme
    write(*,*) '     stopping delta : ',stop_delta
    write(*,*) 'simultaneous search : ',nsearch
    write(*,*)


    o_iter = 0
    mag_dx = 1.0
    do while(mag_dx > stop_delta)
        vars_orig = vars
        i_iter = 0

        write(*,*) 'Constraint Penalties'
        do i=1,nconstraints
            write(*,*) trim(constraintnames(i)%name),' ', penalty(i)
        end do

        write(*,*) 'Beginning New Update Matrix'
        write(*,*) 'iter outer inner Fitness               alpha                 mag(dx)               Vars --> '
        write(*,*) '---- ----- ----- -------------------   -------------------   -------------------   -------------------'
        do while((mag_dx > stop_delta))!.and.(i_iter<1))
            call gradient()
            call append_file(fn,o_iter,i_iter,mag_dx)
            
            if(i_iter .eq. 0) then !set N=identity
                N = 0.0
                do i=1,nvars
                    N(i,i) = 1.0 !N = identity matrix
                end do
            else
                dx(:,1) = vars(:) - vars_old(:)
                gamma(:,1) = grad(:) - grad_old(:)
                NG(:,1) = matmul(N,gamma(:,1))
                denom = dot_product(dx(:,1),gamma(:,1))
        !        N = N + matmul(dx-NG,transpose(dx-NG))/dot_product(dx(:,1)-NG(:,1),gamma(:,1)) !Rank One Hessian Inverse Update
                N = N + (1.0+dot_product(gamma(:,1),NG(:,1))/denom)*(matmul(dx,transpose(dx))/denom) & !BFGS Update
                    & - ( matmul(dx,matmul(transpose(gamma),N)) + matmul(NG,transpose(dx)))/denom
            end if
            s(:) = -matmul(N,grad)
            vars_old = vars
            grad_old = grad
        
            call line_search()

            dx(:,1) = vars(:) - vars_old(:)
            mag_dx = sqrt(dot_product(dx(:,1),dx(:,1)))
            i_iter = i_iter + 1
            iter = iter + 1
        end do

        call append_file(fn,o_iter,i_iter,mag_dx)
        savefile = 'optix_save.json'; call write_optix_file(savefile)

        dx(:,1) = vars(:) - vars_orig(:)
        mag_dx = sqrt(dot_product(dx(:,1),dx(:,1)))
        o_iter = o_iter + 1
        do i=1,nconstraints
            penalty(i) = penalty(i)*penalty_factor(i)
        end do
    end do
    
    call sleep(1)
    fitness_curr = case_fitness_single(0)
!    call start_case(0) !run the final version
    call append_file(fn,o_iter,i_iter,mag_dx)
end subroutine opt_run

!---------------------------------------------------------------------------------------------------------------------------------

subroutine write_optix_file(filename)
    character(100) :: filename,varname
    character(1000) :: runcommands_array
    type(json_value),pointer    :: p_root, p_settings, p_vars, p_vari, p_interface, p_run, p_runi, p_cons
    integer :: iunit,ivar,irun

    !root
    call json_value_create(p_root)           ! create the value and associate the pointer
    call to_object(p_root,trim(filename))    ! add the file name as the name of the overall structure

    !settings structure:
    call json_value_create(p_settings)             !an object
    call to_object(p_settings,'settings')
    call json_value_add(p_root, p_settings)
    call json_value_add(        p_settings, 'default_alpha',         default_alpha)
    call json_value_add(        p_settings, 'diff_delta',            diff_delta)
    call json_value_add(        p_settings, 'diff_scheme',           trim(diff_scheme))
    call json_value_add(        p_settings, 'stop_delta',            stop_delta)
    call json_value_add(        p_settings, 'n_search',              nsearch)
    call json_value_add(        p_settings, 'verbose',               iverbose)
    nullify(p_settings)

    !interface structure:
    call json_value_create(p_interface)             !an object
    call to_object(p_interface,'interface')
    call json_value_add(p_root, p_interface)
    call json_value_add(        p_interface, 'variables',         trim(varfilename))
    call json_value_add(        p_interface, 'fitness',           trim(fitnessfilename))
    nullify(p_interface)

    !run commands structure
    call json_value_create(p_run) !an array
    call to_array(p_run,'run')
    call json_value_add(p_root,p_run)
    do irun=1,nruncommands
        call json_value_create(p_runi)
        call to_string(p_runi,trim(runcommands(irun)%name))
        call json_value_add(p_run,p_runi)
        nullify(p_runi)
    end do
    nullify(p_run)
    
    !variables structure:
    call json_value_create(p_vars)             !an object
    call to_object(p_vars,'variables')
    call json_value_add(p_root, p_vars)

    ivar = 0
    do while (ivar < nvars)
        ivar = ivar + 1
        call json_value_create(p_vari)
        call to_object(p_vari,trim(varnames(ivar)%name))
        call json_value_add(p_vars, p_vari)
        call json_value_add(        p_vari, 'init',            vars(ivar))
        if(opton(ivar).eq.1) call json_value_add(        p_vari, 'opt',     "on")
        if(opton(ivar).eq.0) call json_value_add(        p_vari, 'opt',     "off")
        nullify(p_vari)
    end do
    nullify(p_vars)

    !constraints structure:
    call json_value_create(p_cons)             !an object
    call to_object(p_cons,'constraints')
    call json_value_add(p_root, p_cons)

    ivar = 0
    do while (ivar < nconstraints)
        ivar = ivar + 1
        call json_value_create(p_vari)
        call to_object(p_vari,trim(constraintnames(ivar)%name))
        call json_value_add(p_cons, p_vari)
        if(constrainttype(ivar).eq.1) call json_value_add(        p_vari, 'type',     "=")
        if(constrainttype(ivar).eq.2) call json_value_add(        p_vari, 'type',     "<")
        if(constrainttype(ivar).eq.3) call json_value_add(        p_vari, 'type',     ">")
        call json_value_add(        p_vari, 'value',            constraintvalues(ivar))
        call json_value_add(        p_vari, 'penalty',          penalty(ivar))
        nullify(p_vari)
    end do
    nullify(p_cons)

    !Write File
    open(newunit=iunit, file=filename, status='REPLACE')
    call json_print(p_root,iunit)
    close(iunit)
    call json_destroy(p_root)

end subroutine write_optix_file

!---------------------------------------------------------------------------------------------------------------------------------

subroutine append_file(fn,o_iter,i_iter,mag_dx)
    character(50) :: fn
    real :: mag_dx
    integer :: o_iter,i_iter,ierror
    110 format (3I5, 1000ES22.13)
    write(* ,110) iter,o_iter,i_iter,fitness_curr,alpha,mag_dx,vars(:)
    open(unit = 1001, File = fn, status = "OLD", access = "append", iostat = ierror)
    write(1001,110) iter,o_iter,i_iter,fitness_curr,alpha,mag_dx,vars(:)
    close(1001)
    open(unit = 1001, File = 'gradient.txt', status = "OLD", access = "append", iostat = ierror)
    write(1001,110) iter,o_iter,i_iter,fitness_curr,alpha,mag_dx,grad(:)
    close(1001)
end subroutine append_file

!---------------------------------------------------------------------------------------------------------------------------------

real function case_fitness(case_num,remove)
    integer :: case_num,i
    character(50)::fn,directory,command
    character(1)::remove
    type(json_file) :: f_json    !the JSON structure read from the file
    real :: myvalue
    
    write(directory,*) case_num
    fn = trim(adjustl(directory))//'/'//trim(fitnessfilename)
    call f_json%load_file(filename = fn);       call json_check()

    call f_json%get('fitness',   case_fitness);     call json_check()

    !augment fitness with constraints
    do i=1,nconstraints
        call f_json%get(trim(constraintnames(i)%name),   myvalue);     call json_check()
        select case (constrainttype(i))
            case (1)
                case_fitness = case_fitness + penalty(i)*(constraintvalues(i) - myvalue)**2
            case (2)
                if(myvalue > constraintvalues(i)) then
                    case_fitness = case_fitness + penalty(i)*(constraintvalues(i) - myvalue)**2
                end if
            case (3)
                if(myvalue < constraintvalues(i)) then
                    case_fitness = case_fitness + penalty(i)*(constraintvalues(i) - myvalue)**2
                end if
            case default
        end select
    end do
    
    if(case_fitness .ne. case_fitness) then !NAN returned. Stop program.
        write(*,*) 'NAN found in ',trim(fitnessfilename),' in directory: ',case_num
        write(*,*) 'Ending Program.'
        stop
    end if
    
    if(remove.eq.'y') then
        !delete directory
        command = 'rm -r '//trim(adjustl(directory))
        call system(command)
    end if

end function case_fitness

!---------------------------------------------------------------------------------------------------------------------------------

subroutine forward_diff()
    integer :: i
    real :: vars_orig(nvars)
    vars_orig(:) = vars(:)
    
    grad = 0.0
    call start_case(0)
    do i=1,nvars
        if(opton(i).eq.1) then
            vars(i) = vars(i) + diff_delta
            call start_case(i)
            vars(:) = vars_orig(:)
        end if
    end do    

    do while(.not.all_done())
        call sleep(1)
    end do
    call pause_for_file_write(10) !pause to ensure all files are totally written

    fitness_curr = case_fitness(0,'y')
    do i=1,nvars
        if(opton(i).eq.1) grad(i) = (case_fitness(i,'y') - fitness_curr)/diff_delta
    end do
    
end subroutine forward_diff

!---------------------------------------------------------------------------------------------------------------------------------

subroutine gradient()
    real :: temp(nvars)
    call forward_diff()
    if(trim(diff_scheme).eq.'central') then
        temp(:) = grad(:)
        diff_delta = -diff_delta
        call forward_diff()
        grad(:) = 0.5*(grad(:) + temp(:))
        diff_delta = -diff_delta
    end if
end subroutine gradient

!---------------------------------------------------------------------------------------------------------------------------------

subroutine line_search()
    real :: f1,f2,f3,a1,a2,a3,da,alpha_mult
    real :: xval(0:nsearch),yval(0:nsearch),vars_orig(nvars)
    integer :: j,mincoord
    if(iverbose.eq.1) write(*,*) 'line search ----------------------------------------------------------------------------'

    alpha = max(default_alpha,1.1*stop_delta/sqrt(dot_product(s(:),s(:))))
    vars_orig(:) = vars(:)

    alpha_mult = real(nsearch)/2.0
    xval(0) = 0.0; yval(0) = fitness_curr
    do
        call run_mult_cases(nsearch,alpha,vars_orig,xval(1:nsearch),yval(1:nsearch))
        do j=0,nsearch
            if(iverbose.eq.1) write(*,*) j,xval(j),yval(j)
        end do
        mincoord = minimum_coordinate(nsearch+1,yval)-1
        if(yval(1)>yval(0)) then
            if(alpha*sqrt(dot_product(s(:),s(:))) < stop_delta) then
                write(*,*) 'Line search within stopping tolerance : alpha = ',alpha
                return
            elseif(mincoord.eq.0) then
                if(iverbose.eq.1) write(*,*) 'Too big of a step. Reducing Alpha'
                alpha = alpha/alpha_mult
            else
                if(mincoord.ne.nsearch) exit
                alpha = alpha_mult*alpha
            end if
        else
            if(iverbose.eq.1) write(*,*) 'mincoord = ',mincoord
            if(mincoord.eq.0) return
            if(mincoord.ne.nsearch) exit
            alpha = alpha_mult*alpha
        end if
    end do
    a1 = xval(mincoord-1)
    a2 = xval(mincoord)
    a3 = xval(mincoord+1)
    f1 = yval(mincoord-1)
    f2 = yval(mincoord)
    f3 = yval(mincoord+1)

    da = a2-a1
    alpha = a1+da*(4.0*f2-f3-3.0*f1)/(2.0*(2.0*f2-f3-f1))
    if((alpha > a3).or.(alpha < a1)) then !For parabolas whose min is not in bounds
        alpha = a2
        if(f2 > f1) alpha = a1
    end if
    vars(:) = vars_orig(:) + alpha*s(:)
    call constraints()
    if(iverbose.eq.1) write(*,*) 'final alpha = ',alpha
end subroutine line_search

!---------------------------------------------------------------------------------------------------------------------------------

integer function minimum_coordinate(num,vals)
    integer :: num,i
    real :: vals(num),minval
    minval = vals(1)
    minimum_coordinate = 1
    do i=2,num
        if(vals(i)<minval) then
            minval = vals(i)
            minimum_coordinate = i
        else
            exit !exit at closest minimum in data, not the minimum of the entire data set.
        end if
    end do
end function minimum_coordinate
    
!---------------------------------------------------------------------------------------------------------------------------------

subroutine run_mult_cases(ncases,start_alpha,vars_orig,x,y)
    integer :: ncases,i
    real :: start_alpha,vars_orig(nvars)
    real ::x(ncases),y(ncases)

    do i=1,ncases
        x(i) = real(i)*start_alpha
        vars(:) = vars_orig(:) + x(i)*s(:)
        call start_case(i)
    end do    
    vars(:) = vars_orig(:)
    do while(.not.mult_done(ncases))
        call sleep(1)
    end do
    call pause_for_file_write(10) !pause to ensure all files are totally written

    do i=1,ncases
        y(i) = case_fitness(i,'y')
    end do
    call pause_for_file_write(10) !pause to ensure all files are totally written

end subroutine run_mult_cases

!---------------------------------------------------------------------------------------------------------------------------------

logical function mult_done(ncases)
    implicit none
    integer :: ncases,ios(ncases),i
    character(50)::filename,directory

    do i=1,ncases
        write(directory,*) i
        filename = trim(adjustl(directory))//'/'//trim(fitnessfilename)
        open(i*100,file=filename,status='old',iostat=ios(i))
    end do
    if(count(ios==0)==size(ios)) then
        mult_done = .true.
    else
        mult_done = .false.
    end if
    
    do i=1,ncases
        if(ios(i)/=0) cycle
        close(i*100)
    end do
end function mult_done

!---------------------------------------------------------------------------------------------------------------------------------

real function case_fitness_single(case_num)
    integer :: case_num
    
    call start_case(case_num)
    do while(.not.one_done(case_num))
        call sleep(1)
    end do
    call pause_for_file_write(10) !pause to ensure all files are totally written

    case_fitness_single = case_fitness(case_num,'n')
end function case_fitness_single

!---------------------------------------------------------------------------------------------------------------------------------
subroutine pause_for_file_write(num) !This subroutine forces the operating system to write a file which causes a pause long enough for the blackbox to write the fitness file
    implicit none
    integer :: num,i
    integer :: ierror
    
    do i=1,num
        open(unit = 2000, File = 'pause_file.txt', status="replace", action = "write", iostat = ierror)
        write(2000,*) 0.0
        close(2000)
    end do

end subroutine 
!---------------------------------------------------------------------------------------------------------------------------------

subroutine constraints()
!    integer :: i
    
!    do i=nvars,3*nvars/4+3,-1
!        if(vars(i) < vars(i-1)) then
!            vars(i-1) = vars(i)
!        end if
!    end do
    
end subroutine

!---------------------------------------------------------------------------------------------------------------------------------

subroutine start_case(case_num)
    implicit none
    integer :: case_num,ierror,i
    character(1000)::file_i,command,directory
    real :: temp_vars(nvars),theta
    
    temp_vars(:) = vars
    call constraints()
    
    !Create the directory
    write(directory,*) case_num
    command = 'mkdir '//trim(adjustl(directory))
    call system(command)
    
    !Move to the directory
    call getcwd(command)
    command = trim(adjustl(command))//'/'//trim(adjustl(directory))
    call chdir(command)

    !Write varfilename file to the directory
    open(unit = 10, File = trim(varfilename), status="replace", action = "write", iostat = ierror)
    write(10,*) nvars,' variables'
    do i=1,nvars
        write(10,'(ES25.16)') vars(i)
    end do
    close(10)

    !Run commands within directory
    do i=1,nruncommands
        command = trim(runcommands(i)%name)
        call system(command)
    end do
    
    !move back up a directory
    call chdir('../')
    vars(:) = temp_vars(:)
end subroutine start_case


!---------------------------------------------------------------------------------------------------------------------------------

logical function one_done(case_num)
    implicit none
    integer::case_num,ios
    character(50)::filename,directory
    
    write(directory,*) case_num
    filename = trim(adjustl(directory))//'/'//trim(fitnessfilename)
    open(100,file=filename,status='old',iostat=ios)
    if(ios==0) then
        one_done = .true.
        close(100)
    else
        one_done = .false.
    end if
end function one_done

!---------------------------------------------------------------------------------------------------------------------------------

logical function all_done()
    implicit none
    integer ::ios(nvars+1),i
    character(50)::filename,directory
    
    !Check for 0_ file
    open(100,file='0/'//trim(fitnessfilename),status='old',iostat=ios(nvars+1))
    if(ios(nvars+1)==0) close(100)

    do i=1,nvars
        if(opton(i).eq.1) then
            write(directory,*) i
            filename = trim(adjustl(directory))//'/'//trim(fitnessfilename)
            open(i*100,file=filename,status='old',iostat=ios(i))
        else
            ios(i)=0
        end if
    end do

    if(count(ios==0)==size(ios)) then
        all_done = .true.
    else
        all_done = .false.
    end if
    
    do i=1,nvars
        if(ios(i)/=0) cycle
        close(i*100)
    end do
end function all_done

end module optix
