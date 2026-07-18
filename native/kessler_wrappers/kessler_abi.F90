module pycam_kessler_abi
  use iso_c_binding, only: c_char, c_double, c_f_pointer, c_int, c_null_char, c_ptr
  use state_converters, only: calc_exner_run, temp_to_potential_temp_run
  use state_converters, only: potential_temp_to_temp_run
  use state_converters, only: calc_dry_air_ideal_gas_density_run
  use state_converters, only: wet_to_dry_water_vapor_run
  use state_converters, only: wet_to_dry_cloud_liquid_water_run
  use state_converters, only: wet_to_dry_rain_run
  use state_converters, only: dry_to_wet_water_vapor_run
  use state_converters, only: dry_to_wet_cloud_liquid_water_run
  use state_converters, only: dry_to_wet_rain_run
  use kessler, only: kessler_init, kessler_run
  use kessler_update, only: kessler_update_init, kessler_update_run
  use kessler_update, only: kessler_update_timestep_init, kessler_update_timestep_final
  use check_energy_zero_fluxes, only: check_energy_zero_fluxes_run
  use check_energy_scaling, only: check_energy_scaling_run
  use check_energy_chng, only: check_energy_chng_init, check_energy_chng_run
  use check_energy_chng, only: check_energy_chng_timestep_init
  use dycore_energy_consistency_adjust, only: dycore_energy_consistency_adjust_run
  use physics_tendency_updaters, only: apply_tendency_of_air_temperature_run
  use qneg, only: qneg_init, qneg_run, qneg_timestep_final, qneg_final
  use geopotential_temp, only: geopotential_temp_run
  use ccpp_constituent_prop_mod, only: ccpp_constituent_prop_ptr_t
  use pycam_thermo_water_local, only: thermo_water_update_local
  use kessler_diagnostics, only: native_kessler_diagnostics_init => kessler_diagnostics_init
  use kessler_diagnostics, only: native_kessler_diagnostics_run => kessler_diagnostics_run
  use sima_state_diagnostics, only: native_sima_state_diagnostics_init => sima_state_diagnostics_init
  use sima_state_diagnostics, only: native_sima_state_diagnostics_run => sima_state_diagnostics_run
  use sima_tend_diagnostics, only: native_sima_tend_diagnostics_init => sima_tend_diagnostics_init
  use sima_tend_diagnostics, only: native_sima_tend_diagnostics_run => sima_tend_diagnostics_run
  implicit none
  private

  public :: pycam_kessler_abi_version, pycam_kessler_has_scheme
  public :: pycam_kessler_lifecycle
  public :: pycam_kessler_timestep_initial, pycam_kessler_timestep_final
  public :: pycam_calc_exner_run, pycam_temp_to_potential_temp_run
  public :: pycam_calc_dry_air_ideal_gas_density_run
  public :: pycam_wet_to_dry_water_vapor_run
  public :: pycam_wet_to_dry_cloud_liquid_water_run
  public :: pycam_wet_to_dry_rain_run
  public :: pycam_kessler_run, pycam_potential_temp_to_temp_run
  public :: pycam_dry_to_wet_water_vapor_run
  public :: pycam_dry_to_wet_cloud_liquid_water_run
  public :: pycam_dry_to_wet_rain_run
  public :: pycam_kessler_update_run
  public :: pycam_check_energy_zero_fluxes_run
  public :: pycam_check_energy_scaling_run
  public :: pycam_check_energy_chng_run
  public :: pycam_dycore_energy_consistency_adjust_run
  public :: pycam_apply_tendency_of_air_temperature_run
  public :: pycam_qneg_run, pycam_geopotential_temp_run
  public :: pycam_thermo_water_update_run
  public :: pycam_kessler_diagnostics_run, pycam_sima_state_diagnostics_run
  public :: pycam_sima_tend_diagnostics_run

contains

  integer(c_int) function pycam_kessler_abi_version() bind(C, name="pycam_kessler_abi_version")
    pycam_kessler_abi_version = 1_c_int
  end function pycam_kessler_abi_version

  integer(c_int) function pycam_kessler_has_scheme(name) bind(C, name="pycam_kessler_has_scheme")
    character(kind=c_char), intent(in) :: name(*)
    character(len=96) :: value
    call c_string(name, value)
    select case (trim(value))
    case ("calc_exner", "temp_to_potential_temp", "calc_dry_air_ideal_gas_density", &
          "wet_to_dry_water_vapor", "wet_to_dry_cloud_liquid_water", "wet_to_dry_rain", &
          "kessler", "potential_temp_to_temp", "dry_to_wet_water_vapor", &
          "dry_to_wet_cloud_liquid_water", "dry_to_wet_rain", "kessler_update", &
          "check_energy_zero_fluxes", "check_energy_scaling", "check_energy_chng", &
          "qneg", "geopotential_temp", "thermo_water_update", &
          "dycore_energy_consistency_adjust", "apply_tendency_of_air_temperature", &
          "kessler_diagnostics", "sima_state_diagnostics", "sima_tend_diagnostics")
      pycam_kessler_has_scheme = 1_c_int
    case default
      pycam_kessler_has_scheme = 0_c_int
    end select
  end function pycam_kessler_has_scheme

  integer(c_int) function pycam_kessler_lifecycle(phase, errmsg_c, errmsg_len) &
      bind(C, name="pycam_kessler_lifecycle")
    character(kind=c_char), intent(in) :: phase(*)
    character(kind=c_char), intent(out) :: errmsg_c(*)
    integer(c_int), value :: errmsg_len
    character(len=96) :: value
    character(len=512) :: errmsg
    integer :: errflg
    call c_string(phase, value)
    errmsg = ''
    errflg = 0
    if (trim(value) == "initialize") then
      call kessler_init(2.501e6_c_double, 1.0e5_c_double, 1.0e3_c_double, errmsg, errflg)
      if (errflg == 0) call kessler_update_init(9.80616_c_double, errmsg, errflg)
      if (errflg == 0) call qneg_init('off', 3, errflg, errmsg)
      if (errflg == 0) call check_energy_chng_init(.false.)
      if (errflg == 0) call init_diagnostics(errmsg, errflg)
    else if (trim(value) == "finalize") then
      call finalize_qneg(errmsg, errflg)
    end if
    call copy_error(errmsg, errmsg_c, errmsg_len)
    pycam_kessler_lifecycle = int(errflg, c_int)
  end function pycam_kessler_lifecycle

  integer(c_int) function pycam_kessler_timestep_initial(ncol,nz,ncnst,is_first,q_p,pdel_p,u_p,v_p, &
      temp_p,pintdry_p,phis_p,zm_p,cp_phys_p,cp_dy_p,te_ini_phys_p,te_ini_dyn_p,tw_ini_p, &
      te_cur_phys_p,te_cur_dyn_p,tw_cur_p,tend_te_p,tend_tw_p,temp_ini_p,z_ini_p,count_p,teout_p, &
      energy_phys,energy_dyn,temp_prev_p,ttend_p) bind(C,name="pycam_kessler_timestep_initial")
    integer(c_int),value :: ncol,nz,ncnst,is_first,energy_phys,energy_dyn
    type(c_ptr),value :: q_p,pdel_p,u_p,v_p,temp_p,pintdry_p,phis_p,zm_p,cp_phys_p,cp_dy_p
    type(c_ptr),value :: te_ini_phys_p,te_ini_dyn_p,tw_ini_p,te_cur_phys_p,te_cur_dyn_p,tw_cur_p
    type(c_ptr),value :: tend_te_p,tend_tw_p,temp_ini_p,z_ini_p,count_p,teout_p,temp_prev_p,ttend_p
    real(c_double),pointer :: q(:,:,:),pdel(:,:),u(:,:),v(:,:),temp(:,:),pintdry(:,:),phis(:),zm(:,:)
    real(c_double),pointer :: cp_phys(:,:),cp_dy(:,:),te_ini_phys(:),te_ini_dyn(:),tw_ini(:)
    real(c_double),pointer :: te_cur_phys(:),te_cur_dyn(:),tw_cur(:),tend_te(:),tend_tw(:)
    real(c_double),pointer :: temp_ini(:,:),z_ini(:,:),teout(:),temp_prev(:,:),ttend(:,:)
    integer(c_int),pointer :: count
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(q_p,q,[ncol,nz,ncnst]); call c_f_pointer(pdel_p,pdel,[ncol,nz])
    call c_f_pointer(u_p,u,[ncol,nz]); call c_f_pointer(v_p,v,[ncol,nz]); call c_f_pointer(temp_p,temp,[ncol,nz])
    call c_f_pointer(pintdry_p,pintdry,[ncol,nz+1]); call c_f_pointer(phis_p,phis,[ncol])
    call c_f_pointer(zm_p,zm,[ncol,nz]); call c_f_pointer(cp_phys_p,cp_phys,[ncol,nz])
    call c_f_pointer(cp_dy_p,cp_dy,[ncol,nz]); call c_f_pointer(te_ini_phys_p,te_ini_phys,[ncol])
    call c_f_pointer(te_ini_dyn_p,te_ini_dyn,[ncol]); call c_f_pointer(tw_ini_p,tw_ini,[ncol])
    call c_f_pointer(te_cur_phys_p,te_cur_phys,[ncol]); call c_f_pointer(te_cur_dyn_p,te_cur_dyn,[ncol])
    call c_f_pointer(tw_cur_p,tw_cur,[ncol]); call c_f_pointer(tend_te_p,tend_te,[ncol])
    call c_f_pointer(tend_tw_p,tend_tw,[ncol]); call c_f_pointer(temp_ini_p,temp_ini,[ncol,nz])
    call c_f_pointer(z_ini_p,z_ini,[ncol,nz]); call c_f_pointer(count_p,count)
    call c_f_pointer(teout_p,teout,[ncol]); call c_f_pointer(temp_prev_p,temp_prev,[ncol,nz])
    call c_f_pointer(ttend_p,ttend,[ncol,nz])
    errmsg=''; errflg=0
    call check_energy_chng_timestep_init(ncol,nz,ncnst,is_first /= 0,q,pdel,u,v,temp,pintdry,phis,zm, &
      cp_phys,cp_dy,te_ini_phys,te_ini_dyn,tw_ini,te_cur_phys,te_cur_dyn,tw_cur,tend_te,tend_tw, &
      temp_ini,z_ini,count,teout,energy_phys,energy_dyn,errmsg,errflg)
    if (errflg == 0) call kessler_update_timestep_init(temp,temp_prev,ttend,errmsg,errflg)
    pycam_kessler_timestep_initial=int(errflg,c_int)
  end function pycam_kessler_timestep_initial

  integer(c_int) function pycam_kessler_timestep_final(ncol,nz,cpair_p,temp_p,zm_p,phis_p,st_p) &
      bind(C,name="pycam_kessler_timestep_final")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: cpair_p,temp_p,zm_p,phis_p,st_p
    real(c_double),pointer :: cpair(:,:),temp(:,:),zm(:,:),phis(:),st(:,:)
    type(ccpp_constituent_prop_ptr_t) :: props(3)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(cpair_p,cpair,[ncol,nz]); call c_f_pointer(temp_p,temp,[ncol,nz])
    call c_f_pointer(zm_p,zm,[ncol,nz]); call c_f_pointer(phis_p,phis,[ncol])
    call c_f_pointer(st_p,st,[ncol,nz])
    errmsg=''; errflg=0
    call kessler_update_timestep_final(nz,cpair,temp,zm,phis,st,errflg,errmsg)
    if (errflg == 0) then
      call set_constituent_props(props)
      call qneg_timestep_final(0,0,.true.,6,props,errflg,errmsg)
    end if
    pycam_kessler_timestep_final=int(errflg,c_int)
  end function pycam_kessler_timestep_final

  integer(c_int) function pycam_calc_exner_run(ncol, nz, cpair_p, rair_p, ref_pres, pmid_p, exner_p) &
      bind(C, name="pycam_calc_exner_run")
    integer(c_int), value :: ncol, nz
    type(c_ptr), value :: cpair_p, rair_p, pmid_p, exner_p
    real(c_double), value :: ref_pres
    real(c_double), pointer :: cpair(:,:), rair(:,:), pmid(:,:), exner(:,:)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(cpair_p, cpair, [ncol,nz]); call c_f_pointer(rair_p, rair, [ncol,nz])
    call c_f_pointer(pmid_p, pmid, [ncol,nz]); call c_f_pointer(exner_p, exner, [ncol,nz])
    call calc_exner_run(ncol, nz, cpair, rair, ref_pres, pmid, exner, errmsg, errflg)
    pycam_calc_exner_run = int(errflg, c_int)
  end function pycam_calc_exner_run

  integer(c_int) function pycam_temp_to_potential_temp_run(ncol, nz, temp_p, exner_p, theta_p) &
      bind(C, name="pycam_temp_to_potential_temp_run")
    integer(c_int), value :: ncol, nz
    type(c_ptr), value :: temp_p, exner_p, theta_p
    real(c_double), pointer :: temp(:,:), exner(:,:), theta(:,:)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(temp_p,temp,[ncol,nz]); call c_f_pointer(exner_p,exner,[ncol,nz])
    call c_f_pointer(theta_p,theta,[ncol,nz])
    call temp_to_potential_temp_run(ncol,nz,temp,exner,theta,errmsg,errflg)
    pycam_temp_to_potential_temp_run = int(errflg,c_int)
  end function pycam_temp_to_potential_temp_run

  integer(c_int) function pycam_potential_temp_to_temp_run(ncol, nz, theta_p, exner_p, temp_p) &
      bind(C, name="pycam_potential_temp_to_temp_run")
    integer(c_int), value :: ncol, nz
    type(c_ptr), value :: theta_p, exner_p, temp_p
    real(c_double), pointer :: temp(:,:), exner(:,:), theta(:,:)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(temp_p,temp,[ncol,nz]); call c_f_pointer(exner_p,exner,[ncol,nz])
    call c_f_pointer(theta_p,theta,[ncol,nz])
    call potential_temp_to_temp_run(ncol,nz,theta,exner,temp,errmsg,errflg)
    pycam_potential_temp_to_temp_run = int(errflg,c_int)
  end function pycam_potential_temp_to_temp_run

  integer(c_int) function pycam_calc_dry_air_ideal_gas_density_run(ncol,nz,rair_p,pmiddry_p,temp_p,rho_p) &
      bind(C, name="pycam_calc_dry_air_ideal_gas_density_run")
    integer(c_int), value :: ncol,nz
    type(c_ptr), value :: rair_p,pmiddry_p,temp_p,rho_p
    real(c_double), pointer :: rair(:,:),pmiddry(:,:),temp(:,:),rho(:,:)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(rair_p,rair,[ncol,nz]); call c_f_pointer(pmiddry_p,pmiddry,[ncol,nz])
    call c_f_pointer(temp_p,temp,[ncol,nz]); call c_f_pointer(rho_p,rho,[ncol,nz])
    call calc_dry_air_ideal_gas_density_run(ncol,nz,rair,pmiddry,temp,rho,errmsg,errflg)
    pycam_calc_dry_air_ideal_gas_density_run=int(errflg,c_int)
  end function pycam_calc_dry_air_ideal_gas_density_run

  integer(c_int) function pycam_wet_to_dry_water_vapor_run(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p) &
      bind(C,name="pycam_wet_to_dry_water_vapor_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: pdel_p,pdeldry_p,q_p,qdry_p
    real(c_double),pointer :: pdel(:,:),pdeldry(:,:),q(:,:),qdry(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call ptr4(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p,pdel,pdeldry,q,qdry)
    call wet_to_dry_water_vapor_run(ncol,nz,pdel,pdeldry,q,qdry,errmsg,errflg)
    pycam_wet_to_dry_water_vapor_run=int(errflg,c_int)
  end function

  integer(c_int) function pycam_wet_to_dry_cloud_liquid_water_run(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p) &
      bind(C,name="pycam_wet_to_dry_cloud_liquid_water_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: pdel_p,pdeldry_p,q_p,qdry_p
    real(c_double),pointer :: pdel(:,:),pdeldry(:,:),q(:,:),qdry(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call ptr4(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p,pdel,pdeldry,q,qdry)
    call wet_to_dry_cloud_liquid_water_run(ncol,nz,pdel,pdeldry,q,qdry,errmsg,errflg)
    pycam_wet_to_dry_cloud_liquid_water_run=int(errflg,c_int)
  end function

  integer(c_int) function pycam_wet_to_dry_rain_run(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p) &
      bind(C,name="pycam_wet_to_dry_rain_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: pdel_p,pdeldry_p,q_p,qdry_p
    real(c_double),pointer :: pdel(:,:),pdeldry(:,:),q(:,:),qdry(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call ptr4(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p,pdel,pdeldry,q,qdry)
    call wet_to_dry_rain_run(ncol,nz,pdel,pdeldry,q,qdry,errmsg,errflg)
    pycam_wet_to_dry_rain_run=int(errflg,c_int)
  end function

  integer(c_int) function pycam_dry_to_wet_water_vapor_run(ncol,nz,pdel_p,pdeldry_p,qdry_p,q_p) &
      bind(C,name="pycam_dry_to_wet_water_vapor_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: pdel_p,pdeldry_p,qdry_p,q_p
    real(c_double),pointer :: pdel(:,:),pdeldry(:,:),q(:,:),qdry(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call ptr4(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p,pdel,pdeldry,q,qdry)
    call dry_to_wet_water_vapor_run(ncol,nz,pdel,pdeldry,qdry,q,errmsg,errflg)
    pycam_dry_to_wet_water_vapor_run=int(errflg,c_int)
  end function

  integer(c_int) function pycam_dry_to_wet_cloud_liquid_water_run(ncol,nz,pdel_p,pdeldry_p,qdry_p,q_p) &
      bind(C,name="pycam_dry_to_wet_cloud_liquid_water_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: pdel_p,pdeldry_p,qdry_p,q_p
    real(c_double),pointer :: pdel(:,:),pdeldry(:,:),q(:,:),qdry(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call ptr4(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p,pdel,pdeldry,q,qdry)
    call dry_to_wet_cloud_liquid_water_run(ncol,nz,pdel,pdeldry,qdry,q,errmsg,errflg)
    pycam_dry_to_wet_cloud_liquid_water_run=int(errflg,c_int)
  end function

  integer(c_int) function pycam_dry_to_wet_rain_run(ncol,nz,pdel_p,pdeldry_p,qdry_p,q_p) &
      bind(C,name="pycam_dry_to_wet_rain_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: pdel_p,pdeldry_p,qdry_p,q_p
    real(c_double),pointer :: pdel(:,:),pdeldry(:,:),q(:,:),qdry(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call ptr4(ncol,nz,pdel_p,pdeldry_p,q_p,qdry_p,pdel,pdeldry,q,qdry)
    call dry_to_wet_rain_run(ncol,nz,pdel,pdeldry,qdry,q,errmsg,errflg)
    pycam_dry_to_wet_rain_run=int(errflg,c_int)
  end function

  integer(c_int) function pycam_kessler_run(ncol,nz,dt,lyr_surf,lyr_toa,cpair_p,rair_p,rho_p,z_p,pk_p, &
      theta_p,qv_p,qc_p,qr_p,precl_p,relhum_p) bind(C,name="pycam_kessler_run")
    integer(c_int),value :: ncol,nz,lyr_surf,lyr_toa
    real(c_double),value :: dt
    type(c_ptr),value :: cpair_p,rair_p,rho_p,z_p,pk_p,theta_p,qv_p,qc_p,qr_p,precl_p,relhum_p
    real(c_double),pointer :: cpair(:,:),rair(:,:),rho(:,:),z(:,:),pk(:,:),theta(:,:),qv(:,:),qc(:,:),qr(:,:)
    real(c_double),pointer :: precl(:),relhum(:,:)
    character(len=64) :: scheme_name; character(len=512) :: errmsg; integer :: errflg
    call c_f_pointer(cpair_p,cpair,[ncol,nz]); call c_f_pointer(rair_p,rair,[ncol,nz])
    call c_f_pointer(rho_p,rho,[ncol,nz]); call c_f_pointer(z_p,z,[ncol,nz]); call c_f_pointer(pk_p,pk,[ncol,nz])
    call c_f_pointer(theta_p,theta,[ncol,nz]); call c_f_pointer(qv_p,qv,[ncol,nz]); call c_f_pointer(qc_p,qc,[ncol,nz])
    call c_f_pointer(qr_p,qr,[ncol,nz]); call c_f_pointer(precl_p,precl,[ncol]); call c_f_pointer(relhum_p,relhum,[ncol,nz])
    call kessler_run(ncol,nz,dt,lyr_surf,lyr_toa,cpair,rair,rho,z,pk,theta,qv,qc,qr,precl,relhum, &
      scheme_name,errmsg,errflg)
    pycam_kessler_run=int(errflg,c_int)
  end function pycam_kessler_run

  integer(c_int) function pycam_kessler_update_run(ncol,nz,dt,theta_p,exner_p,temp_prev_p,ttend_p) &
      bind(C,name="pycam_kessler_update_run")
    integer(c_int),value :: ncol,nz
    real(c_double),value :: dt
    type(c_ptr),value :: theta_p,exner_p,temp_prev_p,ttend_p
    real(c_double),pointer :: theta(:,:),exner(:,:),temp_prev(:,:),ttend(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call c_f_pointer(theta_p,theta,[ncol,nz]); call c_f_pointer(exner_p,exner,[ncol,nz])
    call c_f_pointer(temp_prev_p,temp_prev,[ncol,nz]); call c_f_pointer(ttend_p,ttend,[ncol,nz])
    call kessler_update_run(nz,ncol,dt,theta,exner,temp_prev,ttend,errmsg,errflg)
    pycam_kessler_update_run=int(errflg,c_int)
  end function pycam_kessler_update_run

  integer(c_int) function pycam_check_energy_zero_fluxes_run(ncol,flx_vap_p,flx_cnd_p,flx_ice_p,flx_sen_p) &
      bind(C,name="pycam_check_energy_zero_fluxes_run")
    integer(c_int),value :: ncol
    type(c_ptr),value :: flx_vap_p,flx_cnd_p,flx_ice_p,flx_sen_p
    real(c_double),pointer :: flx_vap(:),flx_cnd(:),flx_ice(:),flx_sen(:)
    character(len=64) :: name; character(len=512) :: errmsg; integer :: errflg
    call c_f_pointer(flx_vap_p,flx_vap,[ncol]); call c_f_pointer(flx_cnd_p,flx_cnd,[ncol])
    call c_f_pointer(flx_ice_p,flx_ice,[ncol]); call c_f_pointer(flx_sen_p,flx_sen,[ncol])
    call check_energy_zero_fluxes_run(ncol,name,flx_vap,flx_cnd,flx_ice,flx_sen,errmsg,errflg)
    pycam_check_energy_zero_fluxes_run=int(errflg,c_int)
  end function pycam_check_energy_zero_fluxes_run

  integer(c_int) function pycam_check_energy_scaling_run(ncol,nz,cp_or_cv_p,cpair_p,scaling_p) &
      bind(C,name="pycam_check_energy_scaling_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: cp_or_cv_p,cpair_p,scaling_p
    real(c_double),pointer :: cp_or_cv(:,:),cpair(:,:),scaling(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call c_f_pointer(cp_or_cv_p,cp_or_cv,[ncol,nz]); call c_f_pointer(cpair_p,cpair,[ncol,nz])
    call c_f_pointer(scaling_p,scaling,[ncol,nz])
    call check_energy_scaling_run(ncol,cp_or_cv,cpair,scaling,errmsg,errflg)
    pycam_check_energy_scaling_run=int(errflg,c_int)
  end function pycam_check_energy_scaling_run

  integer(c_int) function pycam_check_energy_chng_run(ncol,nz,ncnst,q_p,pdel_p,u_p,v_p,temp_p, &
      pintdry_p,phis_p,zm_p,cp_phys_p,cp_dy_p,scaling_p,te_cur_phys_p,te_cur_dyn_p,tw_cur_p, &
      tend_te_p,tend_tw_p,temp_ini_p,z_ini_p,count_p,dt,latice,latvap,energy_phys,energy_dyn, &
      flx_vap_p,flx_cnd_p,flx_ice_p,flx_sen_p) bind(C,name="pycam_check_energy_chng_run")
    integer(c_int),value :: ncol,nz,ncnst,energy_phys,energy_dyn
    real(c_double),value :: dt,latice,latvap
    type(c_ptr),value :: q_p,pdel_p,u_p,v_p,temp_p,pintdry_p,phis_p,zm_p,cp_phys_p,cp_dy_p,scaling_p
    type(c_ptr),value :: te_cur_phys_p,te_cur_dyn_p,tw_cur_p,tend_te_p,tend_tw_p,temp_ini_p,z_ini_p,count_p
    type(c_ptr),value :: flx_vap_p,flx_cnd_p,flx_ice_p,flx_sen_p
    real(c_double),pointer :: q(:,:,:),pdel(:,:),u(:,:),v(:,:),temp(:,:),pintdry(:,:),phis(:),zm(:,:)
    real(c_double),pointer :: cp_phys(:,:),cp_dy(:,:),scaling(:,:),te_cur_phys(:),te_cur_dyn(:),tw_cur(:)
    real(c_double),pointer :: tend_te(:),tend_tw(:),temp_ini(:,:),z_ini(:,:)
    real(c_double),pointer :: flx_vap(:),flx_cnd(:),flx_ice(:),flx_sen(:)
    integer(c_int),pointer :: count
    character(len=64) :: name
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(q_p,q,[ncol,nz,ncnst]); call c_f_pointer(pdel_p,pdel,[ncol,nz])
    call c_f_pointer(u_p,u,[ncol,nz]); call c_f_pointer(v_p,v,[ncol,nz]); call c_f_pointer(temp_p,temp,[ncol,nz])
    call c_f_pointer(pintdry_p,pintdry,[ncol,nz+1]); call c_f_pointer(phis_p,phis,[ncol])
    call c_f_pointer(zm_p,zm,[ncol,nz]); call c_f_pointer(cp_phys_p,cp_phys,[ncol,nz])
    call c_f_pointer(cp_dy_p,cp_dy,[ncol,nz]); call c_f_pointer(scaling_p,scaling,[ncol,nz])
    call c_f_pointer(te_cur_phys_p,te_cur_phys,[ncol]); call c_f_pointer(te_cur_dyn_p,te_cur_dyn,[ncol])
    call c_f_pointer(tw_cur_p,tw_cur,[ncol]); call c_f_pointer(tend_te_p,tend_te,[ncol])
    call c_f_pointer(tend_tw_p,tend_tw,[ncol]); call c_f_pointer(temp_ini_p,temp_ini,[ncol,nz])
    call c_f_pointer(z_ini_p,z_ini,[ncol,nz]); call c_f_pointer(count_p,count)
    call c_f_pointer(flx_vap_p,flx_vap,[ncol]); call c_f_pointer(flx_cnd_p,flx_cnd,[ncol])
    call c_f_pointer(flx_ice_p,flx_ice,[ncol]); call c_f_pointer(flx_sen_p,flx_sen,[ncol])
    name='KESSLER'; errmsg=''; errflg=0
    call check_energy_chng_run(ncol,nz,ncnst,6,q,pdel,u,v,temp,pintdry,phis,zm,cp_phys,cp_dy,scaling, &
      te_cur_phys,te_cur_dyn,tw_cur,tend_te,tend_tw,temp_ini,z_ini,count,dt,latice,latvap, &
      energy_phys,energy_dyn,name,flx_vap,flx_cnd,flx_ice,flx_sen,errmsg,errflg)
    pycam_check_energy_chng_run=int(errflg,c_int)
  end function pycam_check_energy_chng_run

  integer(c_int) function pycam_dycore_energy_consistency_adjust_run(ncol,nz,do_adjust,scaling_p,tend_p,local_p) &
      bind(C,name="pycam_dycore_energy_consistency_adjust_run")
    integer(c_int),value :: ncol,nz,do_adjust
    type(c_ptr),value :: scaling_p,tend_p,local_p
    real(c_double),pointer :: scaling(:,:),tend(:,:),local(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call c_f_pointer(scaling_p,scaling,[ncol,nz]); call c_f_pointer(tend_p,tend,[ncol,nz])
    call c_f_pointer(local_p,local,[ncol,nz])
    local=0.0_c_double
    call dycore_energy_consistency_adjust_run(ncol,nz,do_adjust /= 0,scaling,tend,local,errmsg,errflg)
    pycam_dycore_energy_consistency_adjust_run=int(errflg,c_int)
  end function pycam_dycore_energy_consistency_adjust_run

  integer(c_int) function pycam_apply_tendency_of_air_temperature_run(ncol,nz,tend_p,temp_p,total_p,dt) &
      bind(C,name="pycam_apply_tendency_of_air_temperature_run")
    integer(c_int),value :: ncol,nz
    real(c_double),value :: dt
    type(c_ptr),value :: tend_p,temp_p,total_p
    real(c_double),pointer :: tend(:,:),temp(:,:),total(:,:)
    character(len=512) :: errmsg; integer :: errflg
    call c_f_pointer(tend_p,tend,[ncol,nz]); call c_f_pointer(temp_p,temp,[ncol,nz]); call c_f_pointer(total_p,total,[ncol,nz])
    call apply_tendency_of_air_temperature_run(nz,tend,temp,total,dt,errflg,errmsg)
    pycam_apply_tendency_of_air_temperature_run=int(errflg,c_int)
  end function pycam_apply_tendency_of_air_temperature_run

  integer(c_int) function pycam_qneg_run(ncol,nz,ncnst,qmin_p,q_p) bind(C,name="pycam_qneg_run")
    integer(c_int),value :: ncol,nz,ncnst
    type(c_ptr),value :: qmin_p,q_p
    real(c_double),pointer :: qmin(:),q(:,:,:)
    character(len=512) :: errmsg; integer :: errflg
    call c_f_pointer(qmin_p,qmin,[ncnst]); call c_f_pointer(q_p,q,[ncol,nz,ncnst])
    call qneg_run('kessler',ncol,nz,qmin,q,errflg,errmsg)
    pycam_qneg_run=int(errflg,c_int)
  end function pycam_qneg_run

  integer(c_int) function pycam_geopotential_temp_run(ncol,pver,ncnst,lagrang,layer_surf,layer_toa, &
      interface_surf,interface_toa,piln_p,pint_p,pmid_p,pdel_p,rpdel_p,temp_p,qv_p,carr_p,rair_p, &
      gravit,zvir_p,zi_p,zm_p) bind(C,name="pycam_geopotential_temp_run")
    integer(c_int),value :: ncol,pver,ncnst,lagrang,layer_surf,layer_toa,interface_surf,interface_toa
    real(c_double),value :: gravit
    type(c_ptr),value :: piln_p,pint_p,pmid_p,pdel_p,rpdel_p,temp_p,qv_p,carr_p,rair_p,zvir_p,zi_p,zm_p
    real(c_double),pointer :: piln(:,:),pint(:,:),pmid(:,:),pdel(:,:),rpdel(:,:),temp(:,:),qv(:,:)
    real(c_double),pointer :: carr(:,:,:),rair(:,:),zvir(:,:),zi(:,:),zm(:,:)
    type(ccpp_constituent_prop_ptr_t) :: cprops(ncnst)
    character(len=512) :: errmsg; integer :: errflg,i
    call c_f_pointer(piln_p,piln,[ncol,pver+1]); call c_f_pointer(pint_p,pint,[ncol,pver+1])
    call c_f_pointer(pmid_p,pmid,[ncol,pver]); call c_f_pointer(pdel_p,pdel,[ncol,pver])
    call c_f_pointer(rpdel_p,rpdel,[ncol,pver]); call c_f_pointer(temp_p,temp,[ncol,pver])
    call c_f_pointer(qv_p,qv,[ncol,pver]); call c_f_pointer(carr_p,carr,[ncol,pver,ncnst])
    call c_f_pointer(rair_p,rair,[ncol,pver]); call c_f_pointer(zvir_p,zvir,[ncol,pver])
    call c_f_pointer(zi_p,zi,[ncol,pver+1]); call c_f_pointer(zm_p,zm,[ncol,pver])
    do i=1,ncnst
      cprops(i)%thermo_active=.true.; cprops(i)%water_species=.true.
    end do
    call geopotential_temp_run(pver,lagrang/=0,layer_surf,layer_toa,interface_surf,interface_toa,ncnst, &
      piln,pint,pmid,pdel,rpdel,temp,qv,carr,cprops,rair,gravit,zvir,zi,zm,ncol,errflg,errmsg)
    pycam_geopotential_temp_run=int(errflg,c_int)
  end function pycam_geopotential_temp_run

  integer(c_int) function pycam_thermo_water_update_run(ncol,pver,ncnst,mmr_p,pdel_p,pdeldry_p,cpair_p,cpout_p) &
      bind(C,name="pycam_thermo_water_update_run")
    integer(c_int),value :: ncol,pver,ncnst
    type(c_ptr),value :: mmr_p,pdel_p,pdeldry_p,cpair_p,cpout_p
    real(c_double),pointer :: mmr(:,:,:),pdel(:,:),pdeldry(:,:),cpair(:,:),cpout(:,:)
    call c_f_pointer(mmr_p,mmr,[ncol,pver,ncnst]); call c_f_pointer(pdel_p,pdel,[ncol,pver])
    call c_f_pointer(pdeldry_p,pdeldry,[ncol,pver]); call c_f_pointer(cpair_p,cpair,[ncol,pver])
    call c_f_pointer(cpout_p,cpout,[ncol,pver])
    call thermo_water_update_local(mmr,ncol,pver,ncnst,pdel,pdeldry,cpair,cpout)
    pycam_thermo_water_update_run=0_c_int
  end function pycam_thermo_water_update_run

  integer(c_int) function pycam_kessler_diagnostics_run(ncol,precl_p) bind(C,name="pycam_kessler_diagnostics_run")
    integer(c_int),value :: ncol
    type(c_ptr),value :: precl_p
    real(c_double),pointer :: precl(:)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(precl_p,precl,[ncol])
    call native_kessler_diagnostics_run(precl,errmsg,errflg)
    pycam_kessler_diagnostics_run=int(errflg,c_int)
  end function pycam_kessler_diagnostics_run

  integer(c_int) function pycam_sima_state_diagnostics_run(ncol,nz,ncnst,ps_p,psdry_p,phis_p,temp_p,u_p,v_p, &
      dse_p,omega_p,pmid_p,pmiddry_p,pdel_p,pdeldry_p,rpdel_p,rpdeldry_p,lnpmid_p,lnpmiddry_p,inv_exner_p, &
      zm_p,pint_p,pintdry_p,lnpint_p,lnpintdry_p,zi_p,const_p) bind(C,name="pycam_sima_state_diagnostics_run")
    integer(c_int),value :: ncol,nz,ncnst
    type(c_ptr),value :: ps_p,psdry_p,phis_p,temp_p,u_p,v_p,dse_p,omega_p,pmid_p,pmiddry_p,pdel_p,pdeldry_p
    type(c_ptr),value :: rpdel_p,rpdeldry_p,lnpmid_p,lnpmiddry_p,inv_exner_p,zm_p,pint_p,pintdry_p
    type(c_ptr),value :: lnpint_p,lnpintdry_p,zi_p,const_p
    real(c_double),pointer :: ps(:),psdry(:),phis(:),temp(:,:),u(:,:),v(:,:),dse(:,:),omega(:,:)
    real(c_double),pointer :: pmid(:,:),pmiddry(:,:),pdel(:,:),pdeldry(:,:),rpdel(:,:),rpdeldry(:,:)
    real(c_double),pointer :: lnpmid(:,:),lnpmiddry(:,:),inv_exner(:,:),zm(:,:),pint(:,:),pintdry(:,:)
    real(c_double),pointer :: lnpint(:,:),lnpintdry(:,:),zi(:,:),const_array(:,:,:)
    type(ccpp_constituent_prop_ptr_t) :: props(ncnst)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(ps_p,ps,[ncol]); call c_f_pointer(psdry_p,psdry,[ncol]); call c_f_pointer(phis_p,phis,[ncol])
    call c_f_pointer(temp_p,temp,[ncol,nz]); call c_f_pointer(u_p,u,[ncol,nz]); call c_f_pointer(v_p,v,[ncol,nz])
    call c_f_pointer(dse_p,dse,[ncol,nz]); call c_f_pointer(omega_p,omega,[ncol,nz])
    call c_f_pointer(pmid_p,pmid,[ncol,nz]); call c_f_pointer(pmiddry_p,pmiddry,[ncol,nz])
    call c_f_pointer(pdel_p,pdel,[ncol,nz]); call c_f_pointer(pdeldry_p,pdeldry,[ncol,nz])
    call c_f_pointer(rpdel_p,rpdel,[ncol,nz]); call c_f_pointer(rpdeldry_p,rpdeldry,[ncol,nz])
    call c_f_pointer(lnpmid_p,lnpmid,[ncol,nz]); call c_f_pointer(lnpmiddry_p,lnpmiddry,[ncol,nz])
    call c_f_pointer(inv_exner_p,inv_exner,[ncol,nz]); call c_f_pointer(zm_p,zm,[ncol,nz])
    call c_f_pointer(pint_p,pint,[ncol,nz+1]); call c_f_pointer(pintdry_p,pintdry,[ncol,nz+1])
    call c_f_pointer(lnpint_p,lnpint,[ncol,nz+1]); call c_f_pointer(lnpintdry_p,lnpintdry,[ncol,nz+1])
    call c_f_pointer(zi_p,zi,[ncol,nz+1]); call c_f_pointer(const_p,const_array,[ncol,nz,ncnst])
    call set_constituent_props(props)
    call native_sima_state_diagnostics_run(ps,psdry,phis,temp,u,v,dse,omega,pmid,pmiddry,pdel,pdeldry, &
      rpdel,rpdeldry,lnpmid,lnpmiddry,inv_exner,zm,pint,pintdry,lnpint,lnpintdry,zi,const_array,props, &
      errmsg,errflg)
    pycam_sima_state_diagnostics_run=int(errflg,c_int)
  end function pycam_sima_state_diagnostics_run

  integer(c_int) function pycam_sima_tend_diagnostics_run(ncol,nz,ttend_p,utend_p,vtend_p) &
      bind(C,name="pycam_sima_tend_diagnostics_run")
    integer(c_int),value :: ncol,nz
    type(c_ptr),value :: ttend_p,utend_p,vtend_p
    real(c_double),pointer :: ttend(:,:),utend(:,:),vtend(:,:)
    character(len=512) :: errmsg
    integer :: errflg
    call c_f_pointer(ttend_p,ttend,[ncol,nz]); call c_f_pointer(utend_p,utend,[ncol,nz])
    call c_f_pointer(vtend_p,vtend,[ncol,nz])
    call native_sima_tend_diagnostics_run(ttend,utend,vtend,errmsg,errflg)
    pycam_sima_tend_diagnostics_run=int(errflg,c_int)
  end function pycam_sima_tend_diagnostics_run

  subroutine init_diagnostics(errmsg,errflg)
    character(len=512),intent(out) :: errmsg
    integer,intent(out) :: errflg
    type(ccpp_constituent_prop_ptr_t) :: props(3)
    errmsg=''; errflg=0
    call set_constituent_props(props)
    call native_kessler_diagnostics_init(errmsg,errflg)
    if (errflg == 0) call native_sima_state_diagnostics_init(props,errmsg,errflg)
    if (errflg == 0) call native_sima_tend_diagnostics_init(errmsg,errflg)
  end subroutine init_diagnostics

  subroutine finalize_qneg(errmsg,errflg)
    character(len=512),intent(out) :: errmsg
    integer,intent(out) :: errflg
    type(ccpp_constituent_prop_ptr_t) :: props(3)
    call set_constituent_props(props)
    call qneg_final(0,0,.true.,6,props,errflg,errmsg)
  end subroutine finalize_qneg

  subroutine set_constituent_props(props)
    type(ccpp_constituent_prop_ptr_t),intent(out) :: props(:)
    props%thermo_active=.true.
    props%water_species=.true.
    if (size(props) >= 1) props(1)%name='water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water'
    if (size(props) >= 2) props(2)%name='cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water'
    if (size(props) >= 3) props(3)%name='rain_mixing_ratio_wrt_moist_air_and_condensed_water'
  end subroutine set_constituent_props

  subroutine ptr4(ncol,nz,p1,p2,p3,p4,a1,a2,a3,a4)
    integer(c_int),intent(in) :: ncol,nz
    type(c_ptr),value :: p1,p2,p3,p4
    real(c_double),pointer,intent(out) :: a1(:,:),a2(:,:),a3(:,:),a4(:,:)
    call c_f_pointer(p1,a1,[ncol,nz]); call c_f_pointer(p2,a2,[ncol,nz])
    call c_f_pointer(p3,a3,[ncol,nz]); call c_f_pointer(p4,a4,[ncol,nz])
  end subroutine ptr4

  subroutine c_string(source, destination)
    character(kind=c_char), intent(in) :: source(*)
    character(len=*), intent(out) :: destination
    integer :: i
    destination = ''
    do i=1,len(destination)
      if (source(i) == c_null_char) exit
      destination(i:i) = source(i)
    end do
  end subroutine c_string

  subroutine copy_error(source, destination, destination_len)
    character(len=*), intent(in) :: source
    character(kind=c_char), intent(out) :: destination(*)
    integer(c_int), value :: destination_len
    integer :: i, count
    count = min(len_trim(source), int(destination_len)-1)
    do i=1,count
      destination(i)=source(i:i)
    end do
    destination(count+1)=c_null_char
  end subroutine copy_error

end module pycam_kessler_abi
