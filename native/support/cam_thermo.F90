module cam_thermo
  use ccpp_kinds, only: kind_phys
  use cam_thermo_formula, only: ENERGY_FORMULA_DYCORE_FV
  use cam_thermo_formula, only: ENERGY_FORMULA_DYCORE_SE
  use cam_thermo_formula, only: ENERGY_FORMULA_DYCORE_MPAS
  implicit none
  private
  public :: get_hydrostatic_energy

  interface get_hydrostatic_energy
    module procedure get_hydrostatic_energy_1hd
  end interface get_hydrostatic_energy

contains

  subroutine get_hydrostatic_energy_1hd(tracer, moist_mixing_ratio, pdel_in, &
      cp_or_cv, U, V, T, vcoord, ptop, phis, z_mid, dycore_idx, qidx, &
      te, se, po, ke, wv, H2O, liq, ice)
    real(kind_phys), intent(in) :: tracer(:,:,:), pdel_in(:,:), cp_or_cv(:,:)
    real(kind_phys), intent(in) :: U(:,:), V(:,:), T(:,:)
    logical, intent(in) :: moist_mixing_ratio
    integer, intent(in) :: vcoord
    real(kind_phys), intent(in), optional :: ptop(:), phis(:), z_mid(:,:)
    logical, intent(in), optional :: dycore_idx
    integer, intent(in), optional :: qidx
    real(kind_phys), intent(out), optional :: te(:), se(:), po(:), ke(:)
    real(kind_phys), intent(out), optional :: wv(:), H2O(:), liq(:), ice(:)
    real(kind_phys), parameter :: gravit = 9.80616_kind_phys
    real(kind_phys), parameter :: latice = 3.337e5_kind_phys
    real(kind_phys), parameter :: latvap = 2.501e6_kind_phys
    real(kind_phys), parameter :: rga = 1.0_kind_phys / gravit
    real(kind_phys) :: pdel(size(tracer,1),size(tracer,2))
    real(kind_phys) :: ke_vint(size(tracer,1)), se_vint(size(tracer,1))
    real(kind_phys) :: po_vint(size(tracer,1)), wv_vint(size(tracer,1))
    real(kind_phys) :: liq_vint(size(tracer,1)), ice_vint(size(tracer,1))
    integer :: idx, kdx, wvidx

    ! FKESSLER's Python constituent pool is fixed to qv, qc, qr.  This is the
    ! 1-horizontal-dimension algorithm used by CAM-SIMA cam_thermo, kept in
    ! the same loop and arithmetic order without pulling in the full CAM host.
    wvidx = 1
    if (present(qidx)) wvidx = qidx
    pdel = pdel_in
    if (.not. moist_mixing_ratio) then
      do kdx = 1, size(tracer,2)
        do idx = 1, size(tracer,1)
          pdel(idx,kdx) = pdel_in(idx,kdx) * &
              (1.0_kind_phys + sum(tracer(idx,kdx,:)))
        end do
      end do
    end if

    ke_vint = 0.0_kind_phys
    se_vint = 0.0_kind_phys
    select case (vcoord)
    case (ENERGY_FORMULA_DYCORE_FV, ENERGY_FORMULA_DYCORE_SE)
      if (.not. present(ptop) .or. .not. present(phis)) error stop &
          'get_hydrostatic_energy: ptop and phis are required for FV/SE'
      po_vint = ptop
      do kdx = 1, size(tracer,2)
        do idx = 1, size(tracer,1)
          ke_vint(idx) = ke_vint(idx) + pdel(idx,kdx) * 0.5_kind_phys * &
              (U(idx,kdx)**2 + V(idx,kdx)**2) * rga
          se_vint(idx) = se_vint(idx) + T(idx,kdx) * cp_or_cv(idx,kdx) * &
              pdel(idx,kdx) * rga
          po_vint(idx) = po_vint(idx) + pdel(idx,kdx)
        end do
      end do
      do idx = 1, size(tracer,1)
        po_vint(idx) = phis(idx) * po_vint(idx) * rga
      end do
    case (ENERGY_FORMULA_DYCORE_MPAS)
      if (.not. present(phis) .or. .not. present(z_mid)) error stop &
          'get_hydrostatic_energy: phis and z_mid are required for MPAS'
      po_vint = 0.0_kind_phys
      do kdx = 1, size(tracer,2)
        do idx = 1, size(tracer,1)
          ke_vint(idx) = ke_vint(idx) + pdel(idx,kdx) * 0.5_kind_phys * &
              (U(idx,kdx)**2 + V(idx,kdx)**2) * rga
          se_vint(idx) = se_vint(idx) + T(idx,kdx) * cp_or_cv(idx,kdx) * &
              pdel(idx,kdx) * rga
          po_vint(idx) = po_vint(idx) + (z_mid(idx,kdx) + &
              phis(idx) * rga) * pdel(idx,kdx)
        end do
      end do
    case default
      error stop 'get_hydrostatic_energy: unsupported energy formula'
    end select

    wv_vint = 0.0_kind_phys
    liq_vint = 0.0_kind_phys
    ice_vint = 0.0_kind_phys
    if (.not. moist_mixing_ratio) pdel = pdel_in
    do kdx = 1, size(tracer,2)
      do idx = 1, size(tracer,1)
        wv_vint(idx) = wv_vint(idx) + tracer(idx,kdx,wvidx) * &
            pdel(idx,kdx) * rga
        if (size(tracer,3) >= 2) liq_vint(idx) = liq_vint(idx) + &
            tracer(idx,kdx,2) * pdel(idx,kdx) * rga
        if (size(tracer,3) >= 3) liq_vint(idx) = liq_vint(idx) + &
            tracer(idx,kdx,3) * pdel(idx,kdx) * rga
      end do
    end do

    if (present(te)) te = se_vint + po_vint + ke_vint + &
        (latvap + latice) * wv_vint + latice * liq_vint
    if (present(se)) se = se_vint
    if (present(po)) po = po_vint
    if (present(ke)) ke = ke_vint
    if (present(wv)) wv = wv_vint
    if (present(liq)) liq = liq_vint
    if (present(ice)) ice = ice_vint
    if (present(H2O)) H2O = wv_vint + liq_vint + ice_vint
  end subroutine get_hydrostatic_energy_1hd

end module cam_thermo
