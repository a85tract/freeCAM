module pycam_thermo_water_local
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: thermo_water_update_local
contains
  subroutine thermo_water_update_local(mmr, ncol, pver, ncnst, pdel, pdeldry, cpairv, cp_or_cv_dycore)
    real(kind_phys), intent(in) :: mmr(:,:,:)
    integer, intent(in) :: ncol, pver, ncnst
    real(kind_phys), intent(in) :: pdel(:,:), pdeldry(:,:), cpairv(:,:)
    real(kind_phys), intent(out) :: cp_or_cv_dycore(:,:)
    real(kind_phys) :: factor(ncol,pver), sum_species(ncol,pver), sum_cp(ncol,pver)
    real(kind_phys), parameter :: cpwv=1.810e3_kind_phys, cpliq=4.188e3_kind_phys
    integer :: qdx

    factor = pdel(:ncol,:pver) / pdeldry(:ncol,:pver)
    sum_species = 1.0_kind_phys
    do qdx=1,ncnst
      sum_species(:,:) = sum_species(:,:) + mmr(:ncol,:pver,qdx) * factor(:,:)
    end do
    sum_cp = cpairv(:ncol,:pver)
    do qdx=1,ncnst
      if (qdx == 1) then
        sum_cp(:,:) = sum_cp(:,:) + cpwv * mmr(:ncol,:pver,qdx) * factor(:,:)
      else
        sum_cp(:,:) = sum_cp(:,:) + cpliq * mmr(:ncol,:pver,qdx) * factor(:,:)
      end if
    end do
    cp_or_cv_dycore(:ncol,:pver) = sum_cp(:,:) / sum_species(:,:)
  end subroutine thermo_water_update_local
end module pycam_thermo_water_local
