//===-- Test driver for OMEGA Reductions -------------------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA Reductions
///
//
//===-----------------------------------------------------------------------===/

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <mpi.h>

#include "Error.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "Pacer.h"
#include "Reductions.h"

using namespace OMEGA;

// utility function to compute one iteration of the DD algorithm for
// reproducible double precision sums
void sumDDTest(complex<double> &ddb, double &dda) {
   double t1  = dda + real(ddb);
   double e   = t1 - dda;
   double t2  = ((real(ddb) - e) + (dda - (t1 - e))) + imag(ddb);
   ddb = complex<double>(t1 + t2, t2 - ((t1 + t2) - t1));
}

int main(int argc, char *argv[]) {

   // Initialize the global MPI environment
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   Pacer::initialize(MPI_COMM_WORLD);
   Pacer::setPrefix("Omega:");
   {

      // Get some MPI values based on default environment
      MachEnv::init(MPI_COMM_WORLD);
      MachEnv *DefEnv = MachEnv::getDefault();
      MPI_Comm Comm   = DefEnv->getComm();
      int MyTask      = DefEnv->getMyTask();
      int NTasks      = DefEnv->getNumTasks();

      // Initialize the Logging system
      OMEGA::initLogging(DefEnv);
      LOG_INFO("------ Global Reductions Unit Tests ------");

      // For reproducibility across tasks, create a smaller sub-environment
      if (NTasks < 8)
         ABORT_ERROR(
               "ReductionsTest: FAIL must run unit test with at least 8 tasks");
      MachEnv::create("Subset", DefEnv, 4); // contiguous subset environment
      MachEnv *SubEnv  = MachEnv::get("Subset");
      MPI_Comm CommSub = SubEnv->getComm();
      int MyTaskSub    = SubEnv->getMyTask();
      int NTasksSub    = SubEnv->getNumTasks();
      bool IsSubMember = SubEnv->isMember();

      // Set model size for array tests
      int Nx        = 5;
      int NxGlob    = Nx * NTasks;
      int Ny        = 5;
      int Nz        = 3;
      int Nm        = 3;
      int Nn        = 3;
      I8 MaxEntries = NxGlob * Ny * Nz * Nm * Nn;

      // For reproducibility tests, define min, max of all data types
      // We actually want smallest value > 0
      // Further restric the max values so that we do not inadvertantly
      // exceed limits.
      I4 MinI4 = 1;
      I4 MaxI4 = std::numeric_limits<I4>::max()/(10*MaxEntries); 
      I8 MinI8 = 1;
      I8 MaxI8 = std::numeric_limits<I8>::max()/(10*MaxEntries); 
      R4 MinR4 = std::numeric_limits<R4>::min(); 
      R4 MaxR4 = std::numeric_limits<R4>::max()/(10.0*MaxEntries); 
      R8 MinR8 = std::numeric_limits<R8>::min(); 
      R8 MaxR8 = std::numeric_limits<R8>::max()/(10.0*MaxEntries); 
      R4 EpsR4 = 0.0001;
      R8 EpsR8 = 0.0000000000001;
      // Compute max safe exponents based on adjusted max above
      I4 ExpI4 = std::log2(MaxI4);
      I4 ExpI8 = std::log2(MaxI8);
      I4 ExpR4 = std::log10(MaxR4);
      I4 ExpR8 = std::log10(MaxR8);
      // To cover a large range of values, compute a factor for test vals
      // on each rank
      std::vector<I4> FacI4(NTasks);
      std::vector<I8> FacI8(NTasks);
      std::vector<R4> FacR4(NTasks);
      std::vector<R8> FacR8(NTasks);
      for (int I = 0; I < NTasks; ++I) {
         FacI4[I] = pow(2, std::min(I, ExpI4));
         FacI8[I] = pow(2, std::min(I, ExpI8));
         FacR4[I] = pow(10.0, std::min(I, ExpR4));
         FacR8[I] = pow(10.0, std::min(I, ExpR8));
      }

      //------------------------------------------------------------------------
      // Scalar sum sanity checks

      I4 TstI4 = MyTask * FacI4[MyTask];
      I4 SumI4 = 0;
      I4 RefI4 = 0; 
      I4 SubI4 = 0; 
      I8 TstI8 = MyTask * FacI8[MyTask];
      I8 SumI8 = 0;
      I8 RefI8 = 0; 
      I8 SubI8 = 0; 
      R4 TstR4 = (MyTask + EpsR4) * FacR4[MyTask];
      R4 SumR4 = 0.0;
      R4 RefR4 = 0.0; 
      R4 SubR4 = 0.0; 
      R8 TstR8 = (MyTask + EpsR8) * FacR8[MyTask];
      R8 SumR8 = 0.0;
      R8 RefR8 = 0.0; 
      R8 SubR8 = 0.0; 
      R8 TmpR1 = 0.0; // for reproducible R4 sums
      R8 TmpR2 = 0.0; // for reproducible R4 sums

      // for reproducible sums, we need to store running sum and residual as a
      // double complex
      double DDValRef;
      double DDValSub;
      complex<double> DDSumRef(0.0, 0.0);
      complex<double> DDSumSub(0.0, 0.0);
      // Compute reference sums for scalars
      for (int Task = 0; Task < NTasks; ++Task) {
         RefI4 += Task * FacI4[Task];
         RefI8 += Task * FacI8[Task];
         TmpR1 += (Task + EpsR4) * FacR4[Task];
         DDValRef = (Task + EpsR8) * FacR8[Task];
         sumDDTest(DDSumRef, DDValRef);
         if (Task < 4) {
            SubI4 += Task * FacI4[Task];
            SubI8 += Task * FacI8[Task];
            TmpR2 += (Task + EpsR4) * FacR4[Task];
            DDValSub = (Task + EpsR8) * FacR8[Task];
            sumDDTest(DDSumSub, DDValSub);
         }
      }
      RefR4 = TmpR1;
      RefR8 = real(DDSumRef);
      if (MyTask < 4) {
         SubR4 = TmpR2;
         SubR8 = real(DDSumSub);
      }

      // Test global scalar sums against ref values
      SumI4 = globalSum(TstI4, Comm);
      SumI8 = globalSum(TstI8, Comm);
      SumR4 = globalSum(TstR4, Comm);
      SumR8 = globalSum(TstR8, Comm);
      LOG_INFO("Scalar sum I4 {} {}", SumI4, RefI4);
      LOG_INFO("Scalar sum I8 {} {}", SumI8, RefI8);
      LOG_INFO("Scalar sum R4 {} {}", SumR4, RefR4);
      LOG_INFO("Scalar sum R8 {} {}", SumR8, RefR8);
      LOG_ERROR("Barrier");

      if (SumI4 != RefI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (I4 scalar)"
                     "Expected = {} Actual = {}",
                     RefI4, SumI4);
      if (SumI8 != RefI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (I8 scalar)"
                     "Expected = {} Actual = {}",
                     RefI8, SumI8);
      if (SumR4 != RefR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (R4 scalar)"
                     "Expected = {} Actual = {}",
                     RefR4, SumR4);
      if (SumR8 != RefR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (R8 scalar)"
                     "Expected = {} Actual = {}",
                     RefR8, SumR8);

      // Test with subset communicator
      if (IsSubMember) {
         SumI4 = globalSum(TstI4, CommSub);
         SumI8 = globalSum(TstI8, CommSub);
         SumR4 = globalSum(TstR4, CommSub);
         SumR8 = globalSum(TstR8, CommSub);
      LOG_INFO("Scalar sum Sub I4 {} {}", SumI4, SubI4);
      LOG_INFO("Scalar sum Sub I8 {} {}", SumI8, SubI8);
      LOG_INFO("Scalar sum Sub R4 {} {}", SumR4, SubR4);
      LOG_INFO("Scalar sum Sub R8 {} {}", SumR8, SubR8);
      LOG_ERROR("Barrier");

         if (SumI4 != SubI4)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (I4 scalar) subset"
                        "Expected = {} Actual = {}",
                        SubI4, SumI4);
         if (SumI8 != SubI8)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (I8 scalar) subset"
                        "Expected = {} Actual = {}",
                        SubI8, SumI8);
         if (SumR4 != SubR4)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (R4 scalar) subset"
                        "Expected = {} Actual = {}",
                        SubR4, SumR4);
         if (SumR8 != SubR8)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (R8 scalar) subset"
                        "Expected = {} Actual = {}",
                        SubR8, SumR8);
      } // end if subset member

      //------------------------------------------------------------------------
      // Scalar sum multi-field checks - checks both value and reproducibility
      // For reproducibility, we set the second field to have values in the
      // reverse order of the original
      int NFields = 2;
      int NValues = NFields * NTasks;

      std::vector<I4> LocVecI4(NFields);
      std::vector<I8> LocVecI8(NFields);
      std::vector<R4> LocVecR4(NFields);
      std::vector<R8> LocVecR8(NFields);
      std::vector<I4> SumVecI4(NFields);
      std::vector<I8> SumVecI8(NFields);
      std::vector<R4> SumVecR4(NFields);
      std::vector<R8> SumVecR8(NFields);
      I4 TaskRev  = NTasks - 1 - MyTask;
      LocVecI4[0] = TstI4;
      LocVecI8[0] = TstI8;
      LocVecR4[0] = TstR4;
      LocVecR8[0] = TstR8;
      LocVecI4[1] = TaskRev * FacI4[TaskRev];
      LocVecI8[1] = TaskRev * FacI8[TaskRev];
      LocVecR4[1] = (TaskRev + EpsR4) * FacR4[TaskRev];
      LocVecR8[1] = (TaskRev + EpsR8) * FacR8[TaskRev];

      // Test multi-field sums
      SumVecI4 = globalSum(LocVecI4, Comm);
      SumVecI8 = globalSum(LocVecI8, Comm);
      SumVecR4 = globalSum(LocVecR4, Comm);
      SumVecR8 = globalSum(LocVecR8, Comm);
      // Test sum for each field
      for (int I = 0; I < NFields; ++I) {
      LOG_INFO("Scalar sum vect{} I4 {} {}", I, SumVecI4[I], RefI4);
      LOG_INFO("Scalar sum vect{} I8 {} {}", I, SumVecI8[I], RefI8);
      LOG_INFO("Scalar sum vect{} R4 {} {}", I, SumVecR4[I], RefR4);
      LOG_INFO("Scalar sum vect{} R8 {} {}", I, SumVecR8[I], RefR8);
      LOG_ERROR("Barrier");
         if (SumVecI4[I] != RefI4)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (I4 scalar multifield)"
                        "Field {} Expected = {} Actual = {}",
                        I, RefI4, SumVecI4[I]);
         if (SumVecI8[I] != RefI8)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (I8 scalar multifield)"
                        "Field {} Expected = {} Actual = {}",
                        I, RefI8, SumVecI8[I]);
         if (SumVecR4[I] != RefR4)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (R4 scalar multifield)"
                        "Field {} Expected = {} Actual = {}",
                        I, RefR4, SumVecR4[I]);
         if (SumVecR8[I] != RefR8)
            ABORT_ERROR("ReductionsTest: FAIL globalSum (R8 scalar multifield)"
                        "Field {} Expected = {} Actual = {}",
                        I, RefR8, SumVecR8[I]);
      }

      //------------------------------------------------------------------------
      // Full array sum checks

      // Initialize test arrays
      HostArray1DI4 TestHost1DI4("Test1DI4", Nx);
      HostArray2DI4 TestHost2DI4("Test2DI4", Nx, Ny);
      HostArray3DI4 TestHost3DI4("Test3DI4", Nx, Ny, Nz);
      HostArray4DI4 TestHost4DI4("Test4DI4", Nx, Ny, Nz, Nm);
      HostArray5DI4 TestHost5DI4("Test5DI4", Nx, Ny, Nz, Nm, Nn);
      HostArray1DI8 TestHost1DI8("Test1DI8", Nx);
      HostArray2DI8 TestHost2DI8("Test2DI8", Nx, Ny);
      HostArray3DI8 TestHost3DI8("Test3DI8", Nx, Ny, Nz);
      HostArray4DI8 TestHost4DI8("Test4DI8", Nx, Ny, Nz, Nm);
      HostArray5DI8 TestHost5DI8("Test5DI8", Nx, Ny, Nz, Nm, Nn);
      HostArray1DR4 TestHost1DR4("Test1DR4", Nx);
      HostArray2DR4 TestHost2DR4("Test2DR4", Nx, Ny);
      HostArray3DR4 TestHost3DR4("Test3DR4", Nx, Ny, Nz);
      HostArray4DR4 TestHost4DR4("Test4DR4", Nx, Ny, Nz, Nm);
      HostArray5DR4 TestHost5DR4("Test5DR4", Nx, Ny, Nz, Nm, Nn);
      HostArray1DR8 TestHost1DR8("Test1DR8", Nx);
      HostArray2DR8 TestHost2DR8("Test2DR8", Nx, Ny);
      HostArray3DR8 TestHost3DR8("Test3DR8", Nx, Ny, Nz);
      HostArray4DR8 TestHost4DR8("Test4DR8", Nx, Ny, Nz, Nm);
      HostArray5DR8 TestHost5DR8("Test5DR8", Nx, Ny, Nz, Nm, Nn);

      // Compute reference values
      // Float sums must be reproducible
      I4 Ref1DI4 = 0;
      I8 Ref1DI8 = 0;
      R4 Ref1DR4 = 0.0;
      R8 Ref1DR8 = 0.0;
      I4 Ref2DI4 = 0;
      I8 Ref2DI8 = 0;
      R4 Ref2DR4 = 0.0;
      R8 Ref2DR8 = 0.0;
      I4 Ref3DI4 = 0;
      I8 Ref3DI8 = 0;
      R4 Ref3DR4 = 0.0;
      R8 Ref3DR8 = 0.0;
      I4 Ref4DI4 = 0;
      I8 Ref4DI8 = 0;
      R4 Ref4DR4 = 0.0;
      R8 Ref4DR8 = 0.0;
      I4 Ref5DI4 = 0;
      I8 Ref5DI8 = 0;
      R4 Ref5DR4 = 0.0;
      R8 Ref5DR8 = 0.0;
      R8 Tmp1DR4 = 0.0;
      R8 Tmp2DR4 = 0.0;
      R8 Tmp3DR4 = 0.0;
      R8 Tmp4DR4 = 0.0;
      R8 Tmp5DR4 = 0.0;
      R8 DDVal1D = 0.0;
      R8 DDVal2D = 0.0;
      R8 DDVal3D = 0.0;
      R8 DDVal4D = 0.0;
      R8 DDVal5D = 0.0;
      complex<double> DDSumRef1D(0.0, 0.0);
      complex<double> DDSumRef2D(0.0, 0.0);
      complex<double> DDSumRef3D(0.0, 0.0);
      complex<double> DDSumRef4D(0.0, 0.0);
      complex<double> DDSumRef5D(0.0, 0.0);
      // Compute reference sums
      for (int Task = 0; Task < NTasks; ++Task) {
         for (int I = 0; I < Nx; ++I) {
            int IGlob = Task * Nx + I;
            Ref1DI4 += IGlob * FacI4[Task];
            Ref1DI8 += IGlob * FacI8[Task];
            Tmp1DR4 += (IGlob + EpsR4) * FacR4[Task];
            DDVal1D  = (IGlob + EpsR8) * FacR8[Task];
            sumDDTest(DDSumRef1D, DDVal1D); // local repro sum
            for (int J = 0; J < Ny; ++J) {
               int Jindx = IGlob + J;
               Ref2DI4 += Jindx * FacI4[Task];
               Ref2DI8 += Jindx * FacI8[Task];
               Tmp2DR4 += (Jindx + EpsR4) * FacR4[Task];
               DDVal2D  = (Jindx + EpsR8) * FacR8[Task];
               sumDDTest(DDSumRef2D, DDVal2D); // local repro sum
               for (int K = 0; K < Nz; ++K) {
                  int Kindx = IGlob + J + K;
                  Ref3DI4 += Kindx * FacI4[Task];
                  Ref3DI8 += Kindx * FacI8[Task];
                  Tmp3DR4 += (Kindx + EpsR4) * FacR4[Task];
                  DDVal3D  = (Kindx + EpsR8) * FacR8[Task];
                  sumDDTest(DDSumRef3D, DDVal3D); // local repro sum
                  for (int M = 0; M < Nm; ++M) {
                     int Mindx = IGlob + J + K + M;
                     Ref4DI4 += Mindx * FacI4[Task];
                     Ref4DI8 += Mindx * FacI8[Task];
                     Tmp4DR4 += (Mindx + EpsR4) * FacR4[Task];
                     DDVal4D  = (Mindx + EpsR8) * FacR8[Task];
                     sumDDTest(DDSumRef4D, DDVal4D); // local repro sum
                     for (int N = 0; N < Nn; ++N) {
                        int Nindx = IGlob + J + K + M + N;
                        Ref5DI4 += Nindx * FacI4[Task];
                        Ref5DI8 += Nindx * FacI8[Task];
                        Tmp5DR4 += (Nindx + EpsR4) * FacR4[Task];
                        DDVal5D  = (Nindx + EpsR8) * FacR8[Task];
                        sumDDTest(DDSumRef5D, DDVal5D); // local repro sum
                     }
                  }
               }
            }
         }
      }
      Ref1DR4 = Tmp1DR4;
      Ref2DR4 = Tmp2DR4;
      Ref3DR4 = Tmp3DR4;
      Ref4DR4 = Tmp4DR4;
      Ref5DR4 = Tmp5DR4;
      Ref1DR8 = real(DDSumRef1D);
      Ref2DR8 = real(DDSumRef2D);
      Ref3DR8 = real(DDSumRef3D);
      Ref4DR8 = real(DDSumRef4D);
      Ref5DR8 = real(DDSumRef5D);

      // Fill host arrays
      for (int I = 0; I < Nx; ++I) {
         int IGlobal = MyTask * Nx + I;
         TestHost1DI4(I) = IGlobal * FacI4[MyTask];
         TestHost1DI8(I) = IGlobal * FacI8[MyTask];
         TestHost1DR4(I) = (IGlobal + EpsR4) * FacR4[MyTask];
         TestHost1DR8(I) = (IGlobal + EpsR8) * FacR8[MyTask];
         for (int J = 0; J < Ny; ++J) {
            int JIndx = IGlobal + J;
            TestHost2DI4(I, J) = JIndx * FacI4[MyTask];
            TestHost2DI8(I, J) = JIndx * FacI8[MyTask];
            TestHost2DR4(I, J) = (JIndx + EpsR4) * FacR4[MyTask];
            TestHost2DR8(I, J) = (JIndx + EpsR8) * FacR8[MyTask];
            for (int K = 0; K < Nz; ++K) {
               int KIndx = IGlobal + J + K;
               TestHost3DI4(I, J, K) = KIndx * FacI4[MyTask];
               TestHost3DI8(I, J, K) = KIndx * FacI8[MyTask];
               TestHost3DR4(I, J, K) = (KIndx + EpsR4) * FacR4[MyTask];
               TestHost3DR8(I, J, K) = (KIndx + EpsR8) * FacR8[MyTask];
               for (int M = 0; M < Nm; ++M) {
                  int MIndx = IGlobal + J + K + M;
                  TestHost4DI4(I, J, K, M) = MIndx * FacI4[MyTask];
                  TestHost4DI8(I, J, K, M) = MIndx * FacI8[MyTask];
                  TestHost4DR4(I, J, K, M) = (MIndx + EpsR4) * FacR4[MyTask];
                  TestHost4DR8(I, J, K, M) = (MIndx + EpsR8) * FacR8[MyTask];
                  for (int N = 0; N < Nn; ++N) {
                     int NIndx = IGlobal + J + K + M + N;
                     TestHost5DI4(I, J, K, M, N) = NIndx * FacI4[MyTask];
                     TestHost5DI8(I, J, K, M, N) = NIndx * FacI8[MyTask];
                     TestHost5DR4(I, J, K, M, N) = (NIndx + EpsR4) * FacR4[MyTask];
                     TestHost5DR8(I, J, K, M, N) = (NIndx + EpsR8) *
                                                   FacR8[MyTask];
                  }
               }
            }
         }
      }

      // Create device arrays with same values
      auto Test1DI4 = createDeviceMirrorCopy(TestHost1DI4);
      auto Test2DI4 = createDeviceMirrorCopy(TestHost2DI4);
      auto Test3DI4 = createDeviceMirrorCopy(TestHost3DI4);
      auto Test4DI4 = createDeviceMirrorCopy(TestHost4DI4);
      auto Test5DI4 = createDeviceMirrorCopy(TestHost5DI4);
      auto Test1DI8 = createDeviceMirrorCopy(TestHost1DI8);
      auto Test2DI8 = createDeviceMirrorCopy(TestHost2DI8);
      auto Test3DI8 = createDeviceMirrorCopy(TestHost3DI8);
      auto Test4DI8 = createDeviceMirrorCopy(TestHost4DI8);
      auto Test5DI8 = createDeviceMirrorCopy(TestHost5DI8);
      auto Test1DR4 = createDeviceMirrorCopy(TestHost1DR4);
      auto Test2DR4 = createDeviceMirrorCopy(TestHost2DR4);
      auto Test3DR4 = createDeviceMirrorCopy(TestHost3DR4);
      auto Test4DR4 = createDeviceMirrorCopy(TestHost4DR4);
      auto Test5DR4 = createDeviceMirrorCopy(TestHost5DR4);
      auto Test1DR8 = createDeviceMirrorCopy(TestHost1DR8);
      auto Test2DR8 = createDeviceMirrorCopy(TestHost2DR8);
      auto Test3DR8 = createDeviceMirrorCopy(TestHost3DR8);
      auto Test4DR8 = createDeviceMirrorCopy(TestHost4DR8);
      auto Test5DR8 = createDeviceMirrorCopy(TestHost5DR8);

      Kokkos::fence();

      // Compute sums and perform error checks
      SumI4 = globalSum(TestHost1DI4, Comm);
      SumI8 = globalSum(TestHost1DI8, Comm);
      SumR4 = globalSum(TestHost1DR4, Comm);
      SumR8 = globalSum(TestHost1DR8, Comm);
      LOG_INFO("Array1D host sum I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D host sum I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D host sum R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D host sum R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI4 host array)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI8 host array)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR4 host array)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 host array)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      SumI4 = globalSum(Test1DI4, Comm);
      SumI8 = globalSum(Test1DI8, Comm);
      SumR4 = globalSum(Test1DR4, Comm);
      SumR8 = globalSum(Test1DR8, Comm);
      LOG_INFO("Array1D dev sum I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D dev sum I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D dev sum R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D dev sum R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI4 array)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI8 array)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR4 array)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 array)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      SumI4 = globalSum(TestHost2DI4, Comm);
      SumI8 = globalSum(TestHost2DI8, Comm);
      SumR4 = globalSum(TestHost2DR4, Comm);
      SumR8 = globalSum(TestHost2DR8, Comm);
      LOG_INFO("Array2D host sum I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D host sum I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D host sum R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D host sum R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI4 host array)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI8 host array)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR4 host array)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 host array)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      SumI4 = globalSum(Test2DI4, Comm);
      SumI8 = globalSum(Test2DI8, Comm);
      SumR4 = globalSum(Test2DR4, Comm);
      SumR8 = globalSum(Test2DR8, Comm);
      LOG_INFO("Array2D dev sum I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D dev sum I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D dev sum R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D dev sum R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI4 array)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI8 array)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR4 array)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 array)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      SumI4 = globalSum(TestHost3DI4, Comm);
      SumI8 = globalSum(TestHost3DI8, Comm);
      SumR4 = globalSum(TestHost3DR4, Comm);
      SumR8 = globalSum(TestHost3DR8, Comm);
      LOG_INFO("Array3D host sum I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D host sum I8 {} {}", SumI8, Ref3DI8);
      LOG_INFO("Array3D host sum R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D host sum R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI4 host array)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI8 host array)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR4 host array)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 host array)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      SumI4 = globalSum(Test3DI4, Comm);
      SumI8 = globalSum(Test3DI8, Comm);
      SumR4 = globalSum(Test3DR4, Comm);
      SumR8 = globalSum(Test3DR8, Comm);
      LOG_INFO("Array3D dev sum I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D dev sum I8 {} {}", SumI8, Ref3DI8);
      LOG_INFO("Array3D dev sum R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D dev sum R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI4 array)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI8 array)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR4 array)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 array)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      SumI4 = globalSum(TestHost4DI4, Comm);
      SumI8 = globalSum(TestHost4DI8, Comm);
      SumR4 = globalSum(TestHost4DR4, Comm);
      SumR8 = globalSum(TestHost4DR8, Comm);
      LOG_INFO("Array4D host sum I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D host sum I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D host sum R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D host sum R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI4 host array)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI8 host array)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR4 host array)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 host array)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      SumI4 = globalSum(Test4DI4, Comm);
      SumI8 = globalSum(Test4DI8, Comm);
      SumR4 = globalSum(Test4DR4, Comm);
      SumR8 = globalSum(Test4DR8, Comm);
      LOG_INFO("Array4D dev sum I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D dev sum I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D dev sum R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D dev sum R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI4 array)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI8 array)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR4 array)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 array)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      SumI4 = globalSum(TestHost5DI4, Comm);
      SumI8 = globalSum(TestHost5DI8, Comm);
      SumR4 = globalSum(TestHost5DR4, Comm);
      SumR8 = globalSum(TestHost5DR8, Comm);
      LOG_INFO("Array5D host sum I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D host sum I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D host sum R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D host sum R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI4 host array)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI8 host array)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR4 host array)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 host array)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      SumI4 = globalSum(Test5DI4, Comm);
      SumI8 = globalSum(Test5DI8, Comm);
      SumR4 = globalSum(Test5DR4, Comm);
      SumR8 = globalSum(Test5DR8, Comm);
      LOG_INFO("Array5D dev sum I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D dev sum I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D dev sum R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D dev sum R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI4 array)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI8 array)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR4 array)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 array)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      //------------------------------------------------------------------------
      // Test arrays with index range and sum with product by creating a
      // mask that is only 1 where index range is valid and compare.

      // Initialize mask arrays
      HostArray1DI4 MaskHost1DI4("Mask1DI4", Nx);
      HostArray1DI8 MaskHost1DI8("Mask1DI8", Nx);
      HostArray1DR4 MaskHost1DR4("Mask1DR4", Nx);
      HostArray1DR8 MaskHost1DR8("Mask1DR8", Nx);
      HostArray2DI4 MaskHost2DI4("Mask2DI4", Nx, Ny);
      HostArray2DI8 MaskHost2DI8("Mask2DI8", Nx, Ny);
      HostArray2DR4 MaskHost2DR4("Mask2DR4", Nx, Ny);
      HostArray2DR8 MaskHost2DR8("Mask2DR8", Nx, Ny);
      HostArray3DI4 MaskHost3DI4("Mask3DI4", Nx, Ny, Nz);
      HostArray3DI8 MaskHost3DI8("Mask3DI8", Nx, Ny, Nz);
      HostArray3DR4 MaskHost3DR4("Mask3DR4", Nx, Ny, Nz);
      HostArray3DR8 MaskHost3DR8("Mask3DR8", Nx, Ny, Nz);
      HostArray4DI4 MaskHost4DI4("Mask4DI4", Nx, Ny, Nz, Nm);
      HostArray4DI8 MaskHost4DI8("Mask4DI8", Nx, Ny, Nz, Nm);
      HostArray4DR4 MaskHost4DR4("Mask4DR4", Nx, Ny, Nz, Nm);
      HostArray4DR8 MaskHost4DR8("Mask4DR8", Nx, Ny, Nz, Nm);
      HostArray5DI4 MaskHost5DI4("Mask5DI4", Nx, Ny, Nz, Nm, Nn);
      HostArray5DI8 MaskHost5DI8("Mask5DI8", Nx, Ny, Nz, Nm, Nn);
      HostArray5DR4 MaskHost5DR4("Mask5DR4", Nx, Ny, Nz, Nm, Nn);
      HostArray5DR8 MaskHost5DR8("Mask5DR8", Nx, Ny, Nz, Nm, Nn);

      // Reset various sums to compute reference values
      Ref1DI4 = 0;
      Ref1DI8 = 0;
      Ref1DR4 = 0.0;
      Ref1DR8 = 0.0;
      Tmp1DR4 = 0.0;
      DDVal1D = 0.0;
      Ref2DI4 = 0;
      Ref2DI8 = 0;
      Ref2DR4 = 0.0;
      Ref2DR8 = 0.0;
      Tmp2DR4 = 0.0;
      DDVal2D = 0.0;
      Ref3DI4 = 0;
      Ref3DI8 = 0;
      Ref3DR4 = 0.0;
      Ref3DR8 = 0.0;
      Tmp3DR4 = 0.0;
      DDVal3D = 0.0;
      Ref4DI4 = 0;
      Ref4DI8 = 0;
      Ref4DR4 = 0.0;
      Ref4DR8 = 0.0;
      Tmp4DR4 = 0.0;
      DDVal4D = 0.0;
      Ref5DI4 = 0;
      Ref5DI8 = 0;
      Ref5DR4 = 0.0;
      Ref5DR8 = 0.0;
      Tmp5DR4 = 0.0;
      DDVal5D = 0.0;
      DDSumRef1D = complex<double>(0.0, 0.0);
      DDSumRef2D = complex<double>(0.0, 0.0);
      DDSumRef3D = complex<double>(0.0, 0.0);
      DDSumRef4D = complex<double>(0.0, 0.0);
      DDSumRef5D = complex<double>(0.0, 0.0);
      int IMin = 2;
      int IMax = Nx - 2;
      int JMin = 1;
      int JMax = Ny - 1;
      int KMin = 1;
      int KMax = 1;
      int MMin = 1;
      int MMax = 1;
      int NMin = 1;
      int NMax = 1;
      std::vector<int> AddRange(10);
      AddRange[0] = IMin;
      AddRange[1] = IMax;
      AddRange[2] = JMin;
      AddRange[3] = JMax;
      AddRange[4] = KMin;
      AddRange[5] = KMax;
      AddRange[6] = MMin;
      AddRange[7] = MMax;
      AddRange[8] = NMin;
      AddRange[9] = NMax;
      // Compute new reference sums for restricted range
      for (int Task = 0; Task < NTasks; ++Task) {
         for (int I = IMin; I <= IMax; ++I) {
            int IGlob = Task * Nx + I;
            Ref1DI4 += IGlob * FacI4[Task];
            Ref1DI8 += IGlob * FacI8[Task];
            Tmp1DR4 += (IGlob + EpsR4) * FacR4[Task];
            DDVal1D  = (IGlob + EpsR8) * FacR8[Task];
            sumDDTest(DDSumRef1D, DDVal1D); // local repro sum
            for (int J = JMin; J <= JMax; ++J) {
               int Jindx = IGlob + J;
               Ref2DI4 += Jindx * FacI4[Task];
               Ref2DI8 += Jindx * FacI8[Task];
               Tmp2DR4 += (Jindx + EpsR4) * FacR4[Task];
               DDVal2D  = (Jindx + EpsR8) * FacR8[Task];
               sumDDTest(DDSumRef2D, DDVal2D); // local repro sum
               for (int K = KMin; K <= KMax; ++K) {
                  int Kindx = IGlob + J + K;
                  Ref3DI4 += Kindx * FacI4[Task];
                  Ref3DI8 += Kindx * FacI8[Task];
                  Tmp3DR4 += (Kindx + EpsR4) * FacR4[Task];
                  DDVal3D  = (Kindx + EpsR8) * FacR8[Task];
                  sumDDTest(DDSumRef3D, DDVal3D); // local repro sum
                  for (int M = MMin; M <= MMax; ++M) {
                     int Mindx = IGlob + J + K + M;
                     Ref4DI4 += Mindx * FacI4[Task];
                     Ref4DI8 += Mindx * FacI8[Task];
                     Tmp4DR4 += (Mindx + EpsR4) * FacR4[Task];
                     DDVal4D  = (Mindx + EpsR8) * FacR8[Task];
                     sumDDTest(DDSumRef4D, DDVal4D); // local repro sum
                     for (int N = NMin; N <= NMax; ++N) {
                        int Nindx = IGlob + J + K + M + N;
                        Ref5DI4 += Nindx * FacI4[Task];
                        Ref5DI8 += Nindx * FacI8[Task];
                        Tmp5DR4 += (Nindx + EpsR4) * FacR4[Task];
                        DDVal5D  = (Nindx + EpsR8) * FacR8[Task];
                        sumDDTest(DDSumRef5D, DDVal5D); // local repro sum
                     }
                  }
               }
            }
         }
      }
      Ref1DR4 = Tmp1DR4;
      Ref2DR4 = Tmp2DR4;
      Ref3DR4 = Tmp3DR4;
      Ref4DR4 = Tmp4DR4;
      Ref5DR4 = Tmp5DR4;
      Ref1DR8 = real(DDSumRef1D);
      Ref2DR8 = real(DDSumRef2D);
      Ref3DR8 = real(DDSumRef3D);
      Ref4DR8 = real(DDSumRef4D);
      Ref5DR8 = real(DDSumRef5D);

      // Fill mask arrays
      for (int I = IMin; I <= IMax; ++I) {
         MaskHost1DI4(I) = 1;
         MaskHost1DI8(I) = 1;
         MaskHost1DR4(I) = 1.0;
         MaskHost1DR8(I) = 1.0;
         for (int J = JMin; J <= JMax; ++J) {
            MaskHost2DI4(I, J) = 1;
            MaskHost2DI8(I, J) = 1;
            MaskHost2DR4(I, J) = 1.0;
            MaskHost2DR8(I, J) = 1.0;
            for (int K = KMin; K <= KMax; ++K) {
               MaskHost3DI4(I, J, K) = 1;
               MaskHost3DI8(I, J, K) = 1;
               MaskHost3DR4(I, J, K) = 1.0;
               MaskHost3DR8(I, J, K) = 1.0;
               for (int M = MMin; M <= MMax; ++M) {
                  MaskHost4DI4(I, J, K, M) = 1;
                  MaskHost4DI8(I, J, K, M) = 1;
                  MaskHost4DR4(I, J, K, M) = 1.0;
                  MaskHost4DR8(I, J, K, M) = 1.0;
                  for (int N = NMin; N <= NMax; ++N) {
                     MaskHost5DI4(I, J, K, M, N) = 1;
                     MaskHost5DI8(I, J, K, M, N) = 1;
                     MaskHost5DR4(I, J, K, M, N) = 1.0;
                     MaskHost5DR8(I, J, K, M, N) = 1.0;
                  }
               }
            }
         }
      }

      // create equivalent device arrays
      auto Mask1DI4 = createDeviceMirrorCopy(MaskHost1DI4);
      auto Mask1DI8 = createDeviceMirrorCopy(MaskHost1DI8);
      auto Mask1DR4 = createDeviceMirrorCopy(MaskHost1DR4);
      auto Mask1DR8 = createDeviceMirrorCopy(MaskHost1DR8);
      auto Mask2DI4 = createDeviceMirrorCopy(MaskHost2DI4);
      auto Mask2DI8 = createDeviceMirrorCopy(MaskHost2DI8);
      auto Mask2DR4 = createDeviceMirrorCopy(MaskHost2DR4);
      auto Mask2DR8 = createDeviceMirrorCopy(MaskHost2DR8);
      auto Mask3DI4 = createDeviceMirrorCopy(MaskHost3DI4);
      auto Mask3DI8 = createDeviceMirrorCopy(MaskHost3DI8);
      auto Mask3DR4 = createDeviceMirrorCopy(MaskHost3DR4);
      auto Mask3DR8 = createDeviceMirrorCopy(MaskHost3DR8);
      auto Mask4DI4 = createDeviceMirrorCopy(MaskHost4DI4);
      auto Mask4DI8 = createDeviceMirrorCopy(MaskHost4DI8);
      auto Mask4DR4 = createDeviceMirrorCopy(MaskHost4DR4);
      auto Mask4DR8 = createDeviceMirrorCopy(MaskHost4DR8);
      auto Mask5DI4 = createDeviceMirrorCopy(MaskHost5DI4);
      auto Mask5DI8 = createDeviceMirrorCopy(MaskHost5DI8);
      auto Mask5DR4 = createDeviceMirrorCopy(MaskHost5DR4);
      auto Mask5DR8 = createDeviceMirrorCopy(MaskHost5DR8);

      Kokkos::fence();

      //------------------------------------------------------------------------
      // Compute sums with range limits
      SumI4 = globalSum(TestHost1DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost1DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost1DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost1DR8, Comm, &AddRange);
      LOG_INFO("Array1D host sum range I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D host sum range I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D host sum range R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D host sum range R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      SumI4 = globalSum(TestHost2DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost2DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost2DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost2DR8, Comm, &AddRange);
      LOG_INFO("Array2D host sum range I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D host sum range I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D host sum range R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D host sum range R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      SumI4 = globalSum(TestHost3DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost3DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost3DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost3DR8, Comm, &AddRange);
      LOG_INFO("Array3D host sum range I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D host sum range I8 {} {}", SumI8, Ref3DI8);
      LOG_INFO("Array3D host sum range R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D host sum range R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      SumI4 = globalSum(TestHost4DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost4DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost4DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost4DR8, Comm, &AddRange);
      LOG_INFO("Array4D host sum range I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D host sum range I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D host sum range R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D host sum range R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      SumI4 = globalSum(TestHost5DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost5DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost5DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost5DR8, Comm, &AddRange);
      LOG_INFO("Array5D host sum range I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D host sum range I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D host sum range R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D host sum range R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR4 host address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 host address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      SumI4 = globalSum(Test1DI4, Comm, &AddRange);
      SumI8 = globalSum(Test1DI8, Comm, &AddRange);
      SumR4 = globalSum(Test1DR4, Comm, &AddRange);
      SumR8 = globalSum(Test1DR8, Comm, &AddRange);
      LOG_INFO("Array1D dev sum range I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D dev sum range I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D dev sum range R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D dev sum range R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      SumI4 = globalSum(Test2DI4, Comm, &AddRange);
      SumI8 = globalSum(Test2DI8, Comm, &AddRange);
      SumR4 = globalSum(Test2DR4, Comm, &AddRange);
      SumR8 = globalSum(Test2DR8, Comm, &AddRange);
      LOG_INFO("Array2D dev sum range I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D dev sum range I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D dev sum range R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D dev sum range R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      SumI4 = globalSum(Test3DI4, Comm, &AddRange);
      SumI8 = globalSum(Test3DI8, Comm, &AddRange);
      SumR4 = globalSum(Test3DR4, Comm, &AddRange);
      SumR8 = globalSum(Test3DR8, Comm, &AddRange);
      LOG_INFO("Array3D dev sum range I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D dev sum range I8 {} {}", SumI8, Ref3DI8);
      LOG_INFO("Array3D dev sum range R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D dev sum range R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      SumI4 = globalSum(Test4DI4, Comm, &AddRange);
      SumI8 = globalSum(Test4DI8, Comm, &AddRange);
      SumR4 = globalSum(Test4DR4, Comm, &AddRange);
      SumR8 = globalSum(Test4DR8, Comm, &AddRange);
      LOG_INFO("Array4D dev sum range I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D dev sum range I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D dev sum range R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D dev sum range R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      //LOG_ERROR("Before 5d range tests I4");
      SumI4 = globalSum(Test5DI4, Comm, &AddRange);
      //LOG_ERROR("Before 5d range tests I8");
      SumI8 = globalSum(Test5DI8, Comm, &AddRange);
      //LOG_ERROR("Before 5d range tests R4");
      SumR4 = globalSum(Test5DR4, Comm, &AddRange);
      //LOG_ERROR("Before 5d range tests R8");
      SumR8 = globalSum(Test5DR8, Comm, &AddRange);
      LOG_INFO("Array5D dev sum range I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D dev sum range I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D dev sum range R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D dev sum range R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR4 array address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 array address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      //------------------------------------------------------------------------
      // Sum with product - full arrays

      //LOG_ERROR("Before sum with products I4");
      SumI4 = globalSum(TestHost1DI4, MaskHost1DI4, Comm);
      //LOG_ERROR("Before sum with products I8");
      SumI8 = globalSum(TestHost1DI8, MaskHost1DI8, Comm);
      //LOG_ERROR("Before sum with products R4");
      SumR4 = globalSum(TestHost1DR4, MaskHost1DR4, Comm);
      //LOG_ERROR("Before sum with products R8");
      SumR8 = globalSum(TestHost1DR8, MaskHost1DR8, Comm);
      LOG_INFO("Array1D host sum prod I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D host sum prod I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D host sum prod R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D host sum prod R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      //LOG_ERROR("Before sum with products 2D I4");
      SumI4 = globalSum(TestHost2DI4, MaskHost2DI4, Comm);
      //LOG_ERROR("Before sum with products 2D I8");
      SumI8 = globalSum(TestHost2DI8, MaskHost2DI8, Comm);
      //LOG_ERROR("Before sum with products 2D R4");
      SumR4 = globalSum(TestHost2DR4, MaskHost2DR4, Comm);
      //LOG_ERROR("Before sum with products 2D R8");
      SumR8 = globalSum(TestHost2DR8, MaskHost2DR8, Comm);
      LOG_INFO("Array2D host sum prod I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D host sum prod I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D host sum prod R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D host sum prod R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      //LOG_ERROR("Before sum with products 3D");
      SumI4 = globalSum(TestHost3DI4, MaskHost3DI4, Comm);
      SumI8 = globalSum(TestHost3DI8, MaskHost3DI8, Comm);
      SumR4 = globalSum(TestHost3DR4, MaskHost3DR4, Comm);
      SumR8 = globalSum(TestHost3DR8, MaskHost3DR8, Comm);
      LOG_INFO("Array3D host sum prod I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D host sum prod I8 {} {}", SumI8, Ref3DI8);
      LOG_INFO("Array3D host sum prod R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D host sum prod R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      //LOG_ERROR("Before sum with products 4D");
      SumI4 = globalSum(TestHost4DI4, MaskHost4DI4, Comm);
      SumI8 = globalSum(TestHost4DI8, MaskHost4DI8, Comm);
      SumR4 = globalSum(TestHost4DR4, MaskHost4DR4, Comm);
      SumR8 = globalSum(TestHost4DR8, MaskHost4DR8, Comm);
      LOG_INFO("Array4D host sum prod I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D host sum prod I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D host sum prod R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D host sum prod R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      //LOG_ERROR("Before sum with products 5D I4");
      SumI4 = globalSum(TestHost5DI4, MaskHost5DI4, Comm);
      //LOG_ERROR("Before sum with products 5D I8");
      SumI8 = globalSum(TestHost5DI8, MaskHost5DI8, Comm);
      //LOG_ERROR("Before sum with products 5D R4");
      SumR4 = globalSum(TestHost5DR4, MaskHost5DR4, Comm);
      //LOG_ERROR("Before sum with products 5D R8");
      SumR8 = globalSum(TestHost5DR8, MaskHost5DR8, Comm);
      LOG_INFO("Array5D host sum prod I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D host sum prod I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D host sum prod R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D host sum prod R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR4 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 host sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      //LOG_ERROR("Before sum with products dev 1D");
      SumI4 = globalSum(Test1DI4, Mask1DI4, Comm);
      SumI8 = globalSum(Test1DI8, Mask1DI8, Comm);
      SumR4 = globalSum(Test1DR4, Mask1DR4, Comm);
      SumR8 = globalSum(Test1DR8, Mask1DR8, Comm);
      LOG_INFO("Array1D dev sum prod I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D dev sum prod I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D dev sum prod R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D dev sum prod R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DI8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      //LOG_ERROR("Before sum with products dev 2D");
      SumI4 = globalSum(Test2DI4, Mask2DI4, Comm);
      SumI8 = globalSum(Test2DI8, Mask2DI8, Comm);
      SumR4 = globalSum(Test2DR4, Mask2DR4, Comm);
      SumR8 = globalSum(Test2DR8, Mask2DR8, Comm);
      LOG_INFO("Array2D dev sum prod I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D dev sum prod I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D dev sum prod R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D dev sum prod R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DI8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      //LOG_ERROR("Before sum with products dev 3D");
      SumI4 = globalSum(Test3DI4, Mask3DI4, Comm);
      SumI8 = globalSum(Test3DI8, Mask3DI8, Comm);
      SumR4 = globalSum(Test3DR4, Mask3DR4, Comm);
      SumR8 = globalSum(Test3DR8, Mask3DR8, Comm);
      LOG_INFO("Array3D dev sum prod I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D dev sum prod I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array3D dev sum prod R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D dev sum prod R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DI8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      SumI4 = globalSum(Test4DI4, Mask4DI4, Comm);
      SumI8 = globalSum(Test4DI8, Mask4DI8, Comm);
      SumR4 = globalSum(Test4DR4, Mask4DR4, Comm);
      SumR8 = globalSum(Test4DR8, Mask4DR8, Comm);
      LOG_INFO("Array4D dev sum prod I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D dev sum prod I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D dev sum prod R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D dev sum prod R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DI8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      SumI4 = globalSum(Test5DI4, Mask5DI4, Comm);
      SumI8 = globalSum(Test5DI8, Mask5DI8, Comm);
      SumR4 = globalSum(Test5DR4, Mask5DR4, Comm);
      SumR8 = globalSum(Test5DR8, Mask5DR8, Comm);
      LOG_INFO("Array5D dev sum prod I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D dev sum prod I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D dev sum prod R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D dev sum prod R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DI8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR4 sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 sum product)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      //------------------------------------------------------------------------
      // Sum with product - subset range
      SumI4 = globalSum(TestHost1DI4, MaskHost1DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost1DI8, MaskHost1DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost1DR4, MaskHost1DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost1DR8, MaskHost1DR8, Comm, &AddRange);
      LOG_INFO("Array1D host sum prod range I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D host sum prod range I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D host sum prod range R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D host sum prod range R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(1DI4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(1DI8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(1DR4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(1DR8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      SumI4 = globalSum(TestHost2DI4, MaskHost2DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost2DI8, MaskHost2DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost2DR4, MaskHost2DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost2DR8, MaskHost2DR8, Comm, &AddRange);
      LOG_INFO("Array2D host sum prod range I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D host sum prod range I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D host sum prod range R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D host sum prod range R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(2DI4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(2DI8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(2DR4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(2DR8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      SumI4 = globalSum(TestHost3DI4, MaskHost3DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost3DI8, MaskHost3DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost3DR4, MaskHost3DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost3DR8, MaskHost3DR8, Comm, &AddRange);
      LOG_INFO("Array3D host sum prod range I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D host sum prod range I8 {} {}", SumI8, Ref3DI8);
      LOG_INFO("Array3D host sum prod range R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D host sum prod range R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(3DI4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(3DI8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(3DR4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(3DR8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      SumI4 = globalSum(TestHost4DI4, MaskHost4DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost4DI8, MaskHost4DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost4DR4, MaskHost4DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost4DR8, MaskHost4DR8, Comm, &AddRange);
      LOG_INFO("Array4D host sum prod range I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D host sum prod range I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D host sum prod range R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D host sum prod range R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(4DI4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(4DI8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(4DR4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(4DR8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      SumI4 = globalSum(TestHost5DI4, MaskHost5DI4, Comm, &AddRange);
      SumI8 = globalSum(TestHost5DI8, MaskHost5DI8, Comm, &AddRange);
      SumR4 = globalSum(TestHost5DR4, MaskHost5DR4, Comm, &AddRange);
      SumR8 = globalSum(TestHost5DR8, MaskHost5DR8, Comm, &AddRange);
      LOG_INFO("Array5D host sum prod range I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D host sum prod range I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D host sum prod range R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D host sum prod range R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(5DI4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(5DI8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(5DR4 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL "
                     "(5DR8 host sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      SumI4 = globalSum(Test1DI4, Mask1DI4, Comm, &AddRange);
      SumI8 = globalSum(Test1DI8, Mask1DI8, Comm, &AddRange);
      SumR4 = globalSum(Test1DR4, Mask1DR4, Comm, &AddRange);
      SumR8 = globalSum(Test1DR8, Mask1DR8, Comm, &AddRange);
      LOG_INFO("Array1D dev sum prod range I4 {} {}", SumI4, Ref1DI4);
      LOG_INFO("Array1D dev sum prod range I8 {} {}", SumI8, Ref1DI8);
      LOG_INFO("Array1D dev sum prod range R4 {} {}", SumR4, Ref1DR4);
      LOG_INFO("Array1D dev sum prod range R8 {} {}", SumR8, Ref1DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref1DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(1DI4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI4, SumI4);
      if (SumI8 != Ref1DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(1DI8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DI8, SumI8);
      if (SumR4 != Ref1DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(1DR4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR4, SumR4);
      if (SumR8 != Ref1DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(1DR8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref1DR8, SumR8);

      SumI4 = globalSum(Test2DI4, Mask2DI4, Comm, &AddRange);
      SumI8 = globalSum(Test2DI8, Mask2DI8, Comm, &AddRange);
      SumR4 = globalSum(Test2DR4, Mask2DR4, Comm, &AddRange);
      SumR8 = globalSum(Test2DR8, Mask2DR8, Comm, &AddRange);
      LOG_INFO("Array2D dev sum prod range I4 {} {}", SumI4, Ref2DI4);
      LOG_INFO("Array2D dev sum prod range I8 {} {}", SumI8, Ref2DI8);
      LOG_INFO("Array2D dev sum prod range R4 {} {}", SumR4, Ref2DR4);
      LOG_INFO("Array2D dev sum prod range R8 {} {}", SumR8, Ref2DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref2DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(2DI4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI4, SumI4);
      if (SumI8 != Ref2DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(2DI8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DI8, SumI8);
      if (SumR4 != Ref2DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(2DR4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR4, SumR4);
      if (SumR8 != Ref2DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(2DR8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref2DR8, SumR8);

      SumI4 = globalSum(Test3DI4, Mask3DI4, Comm, &AddRange);
      SumI8 = globalSum(Test3DI8, Mask3DI8, Comm, &AddRange);
      SumR4 = globalSum(Test3DR4, Mask3DR4, Comm, &AddRange);
      SumR8 = globalSum(Test3DR8, Mask3DR8, Comm, &AddRange);
      LOG_INFO("Array3D dev sum prod range I4 {} {}", SumI4, Ref3DI4);
      LOG_INFO("Array3D dev sum prod range I8 {} {}", SumI8, Ref3DI8);
      LOG_INFO("Array3D dev sum prod range R4 {} {}", SumR4, Ref3DR4);
      LOG_INFO("Array3D dev sum prod range R8 {} {}", SumR8, Ref3DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref3DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(3DI4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI4, SumI4);
      if (SumI8 != Ref3DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(3DI8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DI8, SumI8);
      if (SumR4 != Ref3DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(3DR4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR4, SumR4);
      if (SumR8 != Ref3DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(3DR8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref3DR8, SumR8);

      SumI4 = globalSum(Test4DI4, Mask4DI4, Comm, &AddRange);
      SumI8 = globalSum(Test4DI8, Mask4DI8, Comm, &AddRange);
      SumR4 = globalSum(Test4DR4, Mask4DR4, Comm, &AddRange);
      SumR8 = globalSum(Test4DR8, Mask4DR8, Comm, &AddRange);
      LOG_INFO("Array4D dev sum prod range I4 {} {}", SumI4, Ref4DI4);
      LOG_INFO("Array4D dev sum prod range I8 {} {}", SumI8, Ref4DI8);
      LOG_INFO("Array4D dev sum prod range R4 {} {}", SumR4, Ref4DR4);
      LOG_INFO("Array4D dev sum prod range R8 {} {}", SumR8, Ref4DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref4DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(4DI4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI4, SumI4);
      if (SumI8 != Ref4DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(4DI8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DI8, SumI8);
      if (SumR4 != Ref4DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(4DR4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR4, SumR4);
      if (SumR8 != Ref4DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(4DR8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref4DR8, SumR8);

      SumI4 = globalSum(Test5DI4, Mask5DI4, Comm, &AddRange);
      SumI8 = globalSum(Test5DI8, Mask5DI8, Comm, &AddRange);
      SumR4 = globalSum(Test5DR4, Mask5DR4, Comm, &AddRange);
      SumR8 = globalSum(Test5DR8, Mask5DR8, Comm, &AddRange);
      LOG_INFO("Array5D dev sum prod range I4 {} {}", SumI4, Ref5DI4);
      LOG_INFO("Array5D dev sum prod range I8 {} {}", SumI8, Ref5DI8);
      LOG_INFO("Array5D dev sum prod range R4 {} {}", SumR4, Ref5DR4);
      LOG_INFO("Array5D dev sum prod range R8 {} {}", SumR8, Ref5DR8);
      LOG_ERROR("Barrier");
      if (SumI4 != Ref5DI4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(5DI4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI4, SumI4);
      if (SumI8 != Ref5DI8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(5DI8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DI8, SumI8);
      if (SumR4 != Ref5DR4)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(5DR4 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR4, SumR4);
      if (SumR8 != Ref5DR8)
         ABORT_ERROR("ReductionsTest: FAIL globalSum "
                     "(5DR8 sum with product and address range)"
                     "Expected = {} Actual = {}",
                     Ref5DR8, SumR8);

      //------------------------------------------------------------------------
      // Multifield sums
      // TODO

      //------------------------------------------------------------------------
      // Multifield sums with subset index range
      // TODO

      //------------------------------------------------------------------------
      // Multifield sums with product
      // TODO

      //------------------------------------------------------------------------
      // Multifield sums with product and subset index range
      // TODO

      //------------------------------------------------------------------------
      // Reproducibility test Array sums
      // TODO

      //------------------------------------------------------------------------
      // Reproducibility test Arrays sums with product
      // TODO

/*
   lsum2 = mpas_globalSum(larray2d, MPI_COMM_WORLD, indxRange)
   dsum3 = mpas_globalSum(darray2d, dmask2d, MPI_COMM_WORLD)
   rsum3 = mpas_globalSum(rarray2d, rmask2d, MPI_COMM_WORLD)
   isum3 = mpas_globalSum(iarray2d, imask2d, MPI_COMM_WORLD)
   lsum3 = mpas_globalSum(larray2d, lmask2d, MPI_COMM_WORLD)
   dsum4 = mpas_globalSum(darray2d, dmask2d, MPI_COMM_WORLD, indxRange)
   rsum4 = mpas_globalSum(rarray2d, rmask2d, MPI_COMM_WORLD, indxRange)
   isum4 = mpas_globalSum(iarray2d, imask2d, MPI_COMM_WORLD, indxRange)
   lsum4 = mpas_globalSum(larray2d, lmask2d, MPI_COMM_WORLD, indxRange)
   if (dsum1 == dref) then
      print *, 'Global sum array2d double: PASS'
   else
      print *, 'Global sum array2d double: FAIL', dsum1, dref
   endif
   if (rsum1 == rref) then
      print *, 'Global sum array2d real: PASS'
   else
      print *, 'Global sum array2d real: FAIL', rsum1, rref
   endif
   if (isum1 == iref) then
      print *, 'Global sum array2d integer: PASS'
   else
      print *, 'Global sum array2d integer: FAIL', isum1, iref
   endif
   if (lsum1 == lref) then
      print *, 'Global sum array2d integer(8): PASS'
   else
      print *, 'Global sum array2d integer(8): FAIL', lsum1, lref
   endif
   if (dsum2 == dref2) then
      print *, 'Global sum irange array2d double: PASS'
   else
      print *, 'Global sum irange array2d double: FAIL', dsum2, dref2
   endif
   if (rsum2 == rref2) then
      print *, 'Global sum irange array2d real: PASS'
   else
      print *, 'Global sum irange array2d real: FAIL', rsum2, rref2
   endif
   if (isum2 == iref2) then
      print *, 'Global sum irange array2d integer: PASS'
   else
      print *, 'Global sum irange array2d integer: FAIL', isum2, iref2
   endif
   if (lsum2 == lref2) then
      print *, 'Global sum irange array2d integer(8): PASS'
   else
      print *, 'Global sum irange array2d integer(8): FAIL', lsum2, lref2
   endif
   if (dsum3 == dref2) then
      print *, 'Global sum prod array2d double: PASS'
   else
      print *, 'Global sum prod array2d double: FAIL', dsum3, dref2
   endif
   if (rsum3 == rref2) then
      print *, 'Global sum prod array2d real: PASS'
   else
      print *, 'Global sum prod array2d real: FAIL', rsum3, rref2
   endif
   if (isum3 == iref2) then
      print *, 'Global sum prod array2d integer: PASS'
   else
      print *, 'Global sum prod array2d integer: FAIL', isum3, iref2
   endif
   if (lsum3 == lref2) then
      print *, 'Global sum prod array2d integer(8): PASS'
   else
      print *, 'Global sum prod array2d integer(8): FAIL', lsum3, lref2
   endif
   if (dsum4 == dref2) then
      print *, 'Global sum prod irange array2d double: PASS'
   else
      print *, 'Global sum prod irange array2d double: FAIL', dsum4, dref2
   endif
   if (rsum4 == rref2) then
      print *, 'Global sum prod irange array2d real: PASS'
   else
      print *, 'Global sum prod irange array2d real: FAIL', rsum4, rref2
   endif
   if (isum4 == iref2) then
      print *, 'Global sum prod irange array2d integer: PASS'
   else
      print *, 'Global sum prod irange array2d integer: FAIL', isum4, iref2
   endif
   if (lsum4 == lref2) then
      print *, 'Global sum prod irange array2d integer(8): PASS'
   else
      print *, 'Global sum prod irange array2d integer(8): FAIL', lsum4, lref2
   endif
   deallocate(darray2d, dmask2d, rarray2d, rmask2d, &
              iarray2d, imask2d, larray2d, lmask2d)


   !*** 3-d arrays
   if (myRank == masterTask) print *,'Testing 3d sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray3d(nx,ny,nz), dmask3d(nx,ny,nz), &
            rarray3d(nx,ny,nz), rmask3d(nx,ny,nz), &
            iarray3d(nx,ny,nz), imask3d(nx,ny,nz), &
            larray3d(nx,ny,nz), lmask3d(nx,ny,nz))
   dref = 0.d0
   rref = 0.0
   iref = 0
   lref = 0
   dref2 = 0.d0
   rref2 = 0.0
   iref2 = 0
   lref2 = 0
   do k=1,nz
   do j=1,ny
   do i=1,nx
      lscalar = (i+j+k)*(myRank+1)
      darray3d(i,j,k) = real(lscalar,R8)
      rarray3d(i,j,k) = real(lscalar)
      larray3d(i,j,k) = lscalar
      iarray3d(i,j,k) = min(lscalar,maxint)
      do r=0,nRanks-1
         dref = dref + real((i+j+k)*(r+1),R8)
         lref = lref + (i+j+k)*(r+1)
      end do
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4) .and. &
          k >= indxRange(5) .and. k <= indxRange(6)) then
         dmask3d(i,j,k) = 1.d0
         rmask3d(i,j,k) = 1.0
         lmask3d(i,j,k) = 1
         imask3d(i,j,k) = 1
         do r=0,nRanks-1
            dref2 = dref2 + real((i+j+k)*(r+1),R8)
            lref2 = lref2 + (i+j+k)*(r+1)
         end do
      else
         dmask3d(i,j,k) = 0.d0
         rmask3d(i,j,k) = 0.0
         lmask3d(i,j,k) = 0
         imask3d(i,j,k) = 0
      endif
   end do
   end do
   end do
   rref  = dref
   rref2 = dref2
   iref  = lref
   iref2 = lref2
   dsum1 = mpas_globalSum(darray3d, MPI_COMM_WORLD)
   rsum1 = mpas_globalSum(rarray3d, MPI_COMM_WORLD)
   isum1 = mpas_globalSum(iarray3d, MPI_COMM_WORLD)
   lsum1 = mpas_globalSum(larray3d, MPI_COMM_WORLD)
   dsum2 = mpas_globalSum(darray3d, MPI_COMM_WORLD, indxRange)
   rsum2 = mpas_globalSum(rarray3d, MPI_COMM_WORLD, indxRange)
   isum2 = mpas_globalSum(iarray3d, MPI_COMM_WORLD, indxRange)
   lsum2 = mpas_globalSum(larray3d, MPI_COMM_WORLD, indxRange)
   dsum3 = mpas_globalSum(darray3d, dmask3d, MPI_COMM_WORLD)
   rsum3 = mpas_globalSum(rarray3d, rmask3d, MPI_COMM_WORLD)
   isum3 = mpas_globalSum(iarray3d, imask3d, MPI_COMM_WORLD)
   lsum3 = mpas_globalSum(larray3d, lmask3d, MPI_COMM_WORLD)
   dsum4 = mpas_globalSum(darray3d, dmask3d, MPI_COMM_WORLD, indxRange)
   rsum4 = mpas_globalSum(rarray3d, rmask3d, MPI_COMM_WORLD, indxRange)
   isum4 = mpas_globalSum(iarray3d, imask3d, MPI_COMM_WORLD, indxRange)
   lsum4 = mpas_globalSum(larray3d, lmask3d, MPI_COMM_WORLD, indxRange)
   if (dsum1 == dref) then
      print *, 'Global sum array3d double: PASS'
   else
      print *, 'Global sum array3d double: FAIL', dsum1, dref
   endif
   if (rsum1 == rref) then
      print *, 'Global sum array3d real: PASS'
   else
      print *, 'Global sum array3d real: FAIL', rsum1, rref
   endif
   if (isum1 == iref) then
      print *, 'Global sum array3d integer: PASS'
   else
      print *, 'Global sum array3d integer: FAIL', isum1, iref
   endif
   if (lsum1 == lref) then
      print *, 'Global sum array3d integer(8): PASS'
   else
      print *, 'Global sum array3d integer(8): FAIL', lsum1, lref
   endif
   if (dsum2 == dref2) then
      print *, 'Global sum irange array3d double: PASS'
   else
      print *, 'Global sum irange array3d double: FAIL', dsum2, dref2
   endif
   if (rsum2 == rref2) then
      print *, 'Global sum irange array3d real: PASS'
   else
      print *, 'Global sum irange array3d real: FAIL', rsum2, rref2
   endif
   if (isum2 == iref2) then
      print *, 'Global sum irange array3d integer: PASS'
   else
      print *, 'Global sum irange array3d integer: FAIL', isum2, iref2
   endif
   if (lsum2 == lref2) then
      print *, 'Global sum irange array3d integer(8): PASS'
   else
      print *, 'Global sum irange array3d integer(8): FAIL', lsum2, lref2
   endif
   if (dsum3 == dref2) then
      print *, 'Global sum prod array3d double: PASS'
   else
      print *, 'Global sum prod array3d double: FAIL', dsum3, dref2
   endif
   if (rsum3 == rref2) then
      print *, 'Global sum prod array3d real: PASS'
   else
      print *, 'Global sum prod array3d real: FAIL', rsum3, rref2
   endif
   if (isum3 == iref2) then
      print *, 'Global sum prod array3d integer: PASS'
   else
      print *, 'Global sum prod array3d integer: FAIL', isum3, iref2
   endif
   if (lsum3 == lref2) then
      print *, 'Global sum prod array3d integer(8): PASS'
   else
      print *, 'Global sum prod array3d integer(8): FAIL', lsum3, lref2
   endif
   if (dsum4 == dref2) then
      print *, 'Global sum prod irange array3d double: PASS'
   else
      print *, 'Global sum prod irange array3d double: FAIL', dsum4, dref2
   endif
   if (rsum4 == rref2) then
      print *, 'Global sum prod irange array3d real: PASS'
   else
      print *, 'Global sum prod irange array3d real: FAIL', rsum4, rref2
   endif
   if (isum4 == iref2) then
      print *, 'Global sum prod irange array3d integer: PASS'
   else
      print *, 'Global sum prod irange array3d integer: FAIL', isum4, iref2
   endif
   if (lsum4 == lref2) then
      print *, 'Global sum prod irange array3d integer(8): PASS'
   else
      print *, 'Global sum prod irange array3d integer(8): FAIL', lsum4, lref2
   endif
   deallocate(darray3d, dmask3d, rarray3d, rmask3d, &
              iarray3d, imask3d, larray3d, lmask3d)

   !*** 4-d arrays
   if (myRank == masterTask) print *,'Testing 4d sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray4d(nx,ny,nz,nl), dmask4d(nx,ny,nz,nl), &
            rarray4d(nx,ny,nz,nl), rmask4d(nx,ny,nz,nl), &
            iarray4d(nx,ny,nz,nl), imask4d(nx,ny,nz,nl), &
            larray4d(nx,ny,nz,nl), lmask4d(nx,ny,nz,nl))
   dref = 0.d0
   rref = 0.0
   iref = 0
   lref = 0
   dref2 = 0.d0
   rref2 = 0.0
   iref2 = 0
   lref2 = 0
   do l=1,nl
   do k=1,nz
   do j=1,ny
   do i=1,nx
      lscalar = (i+j+k+l)*(myRank+1)
      darray4d(i,j,k,l) = real(lscalar,R8)
      rarray4d(i,j,k,l) = real(lscalar)
      larray4d(i,j,k,l) = lscalar
      iarray4d(i,j,k,l) = min(lscalar,maxint)
      do r=0,nRanks-1
         dref = dref + real((i+j+k+l)*(r+1),R8)
         lref = lref + (i+j+k+l)*(r+1)
      end do
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4) .and. &
          k >= indxRange(5) .and. k <= indxRange(6) .and. &
          l >= indxRange(7) .and. l <= indxRange(8)) then
         dmask4d(i,j,k,l) = 1.d0
         rmask4d(i,j,k,l) = 1.0
         lmask4d(i,j,k,l) = 1
         imask4d(i,j,k,l) = 1
         do r=0,nRanks-1
            dref2 = dref2 + real((i+j+k+l)*(r+1),R8)
            lref2 = lref2 + (i+j+k+l)*(r+1)
         end do
      else
         dmask4d(i,j,k,l) = 0.d0
         rmask4d(i,j,k,l) = 0.0
         lmask4d(i,j,k,l) = 0
         imask4d(i,j,k,l) = 0
      endif
   end do
   end do
   end do
   end do
   rref  = dref
   rref2 = dref2
   iref  = lref
   iref2 = lref2
   dsum1 = mpas_globalSum(darray4d, MPI_COMM_WORLD)
   rsum1 = mpas_globalSum(rarray4d, MPI_COMM_WORLD)
   isum1 = mpas_globalSum(iarray4d, MPI_COMM_WORLD)
   lsum1 = mpas_globalSum(larray4d, MPI_COMM_WORLD)
   dsum2 = mpas_globalSum(darray4d, MPI_COMM_WORLD, indxRange)
   rsum2 = mpas_globalSum(rarray4d, MPI_COMM_WORLD, indxRange)
   isum2 = mpas_globalSum(iarray4d, MPI_COMM_WORLD, indxRange)
   lsum2 = mpas_globalSum(larray4d, MPI_COMM_WORLD, indxRange)
   dsum3 = mpas_globalSum(darray4d, dmask4d, MPI_COMM_WORLD)
   rsum3 = mpas_globalSum(rarray4d, rmask4d, MPI_COMM_WORLD)
   isum3 = mpas_globalSum(iarray4d, imask4d, MPI_COMM_WORLD)
   lsum3 = mpas_globalSum(larray4d, lmask4d, MPI_COMM_WORLD)
   dsum4 = mpas_globalSum(darray4d, dmask4d, MPI_COMM_WORLD, indxRange)
   rsum4 = mpas_globalSum(rarray4d, rmask4d, MPI_COMM_WORLD, indxRange)
   isum4 = mpas_globalSum(iarray4d, imask4d, MPI_COMM_WORLD, indxRange)
   lsum4 = mpas_globalSum(larray4d, lmask4d, MPI_COMM_WORLD, indxRange)
   if (dsum1 == dref) then
      print *, 'Global sum array4d double: PASS'
   else
      print *, 'Global sum array4d double: FAIL', dsum1, dref
   endif
   if (rsum1 == rref) then
      print *, 'Global sum array4d real: PASS'
   else
      print *, 'Global sum array4d real: FAIL', rsum1, rref
   endif
   if (isum1 == iref) then
      print *, 'Global sum array4d integer: PASS'
   else
      print *, 'Global sum array4d integer: FAIL', isum1, iref
   endif
   if (lsum1 == lref) then
      print *, 'Global sum array4d integer(8): PASS'
   else
      print *, 'Global sum array4d integer(8): FAIL', lsum1, lref
   endif
   if (dsum2 == dref2) then
      print *, 'Global sum irange array4d double: PASS'
   else
      print *, 'Global sum irange array4d double: FAIL', dsum2, dref2
   endif
   if (rsum2 == rref2) then
      print *, 'Global sum irange array4d real: PASS'
   else
      print *, 'Global sum irange array4d real: FAIL', rsum2, rref2
   endif
   if (isum2 == iref2) then
      print *, 'Global sum irange array4d integer: PASS'
   else
      print *, 'Global sum irange array4d integer: FAIL', isum2, iref2
   endif
   if (lsum2 == lref2) then
      print *, 'Global sum irange array4d integer(8): PASS'
   else
      print *, 'Global sum irange array4d integer(8): FAIL', lsum2, lref2
   endif
   if (dsum3 == dref2) then
      print *, 'Global sum prod array4d double: PASS'
   else
      print *, 'Global sum prod array4d double: FAIL', dsum3, dref2
   endif
   if (rsum3 == rref2) then
      print *, 'Global sum prod array4d real: PASS'
   else
      print *, 'Global sum prod array4d real: FAIL', rsum3, rref2
   endif
   if (isum3 == iref2) then
      print *, 'Global sum prod array4d integer: PASS'
   else
      print *, 'Global sum prod array4d integer: FAIL', isum3, iref2
   endif
   if (lsum3 == lref2) then
      print *, 'Global sum prod array4d integer(8): PASS'
   else
      print *, 'Global sum prod array4d integer(8): FAIL', lsum3, lref2
   endif
   if (dsum4 == dref2) then
      print *, 'Global sum prod irange array4d double: PASS'
   else
      print *, 'Global sum prod irange array4d double: FAIL', dsum4, dref2
   endif
   if (rsum4 == rref2) then
      print *, 'Global sum prod irange array4d real: PASS'
   else
      print *, 'Global sum prod irange array4d real: FAIL', rsum4, rref2
   endif
   if (isum4 == iref2) then
      print *, 'Global sum prod irange array4d integer: PASS'
   else
      print *, 'Global sum prod irange array4d integer: FAIL', isum4, iref2
   endif
   if (lsum4 == lref2) then
      print *, 'Global sum prod irange array4d integer(8): PASS'
   else
      print *, 'Global sum prod irange array4d integer(8): FAIL', lsum4, lref2
   endif
   deallocate(darray4d, dmask4d, rarray4d, rmask4d, &
              iarray4d, imask4d, larray4d, lmask4d)


   !*** 5-d arrays
   if (myRank == masterTask) print *,'Testing 5d sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray5d(nx,ny,nz,nl,nm), dmask5d(nx,ny,nz,nl,nm), &
            rarray5d(nx,ny,nz,nl,nm), rmask5d(nx,ny,nz,nl,nm), &
            iarray5d(nx,ny,nz,nl,nm), imask5d(nx,ny,nz,nl,nm), &
            larray5d(nx,ny,nz,nl,nm), lmask5d(nx,ny,nz,nl,nm))
   dref = 0.d0
   rref = 0.0
   iref = 0
   lref = 0
   dref2 = 0.d0
   rref2 = 0.0
   iref2 = 0
   lref2 = 0
   do m=1,nm
   do l=1,nl
   do k=1,nz
   do j=1,ny
   do i=1,nx
      lscalar = (i+j+k+l+m)*(myRank+1)
      darray5d(i,j,k,l,m) = real(lscalar,R8)
      rarray5d(i,j,k,l,m) = real(lscalar)
      larray5d(i,j,k,l,m) = lscalar
      iarray5d(i,j,k,l,m) = min(lscalar,maxint)
      do r=0,nRanks-1
         dref = dref + real((i+j+k+l+m)*(r+1),R8)
         lref = lref + (i+j+k+l+m)*(r+1)
      end do
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4) .and. &
          k >= indxRange(5) .and. k <= indxRange(6) .and. &
          l >= indxRange(7) .and. l <= indxRange(8) .and. &
          m >= indxRange(9) .and. m <= indxRange(10)) then
         dmask5d(i,j,k,l,m) = 1.d0
         rmask5d(i,j,k,l,m) = 1.0
         lmask5d(i,j,k,l,m) = 1
         imask5d(i,j,k,l,m) = 1
         do r=0,nRanks-1
            dref2 = dref2 + real((i+j+k+l+m)*(r+1),R8)
            lref2 = lref2 + (i+j+k+l+m)*(r+1)
         end do
      else
         dmask5d(i,j,k,l,m) = 0.d0
         rmask5d(i,j,k,l,m) = 0.0
         lmask5d(i,j,k,l,m) = 0
         imask5d(i,j,k,l,m) = 0
      endif
   end do
   end do
   end do
   end do
   end do
   rref  = dref
   rref2 = dref2
   iref  = lref
   iref2 = lref2
   dsum1 = mpas_globalSum(darray5d, MPI_COMM_WORLD)
   rsum1 = mpas_globalSum(rarray5d, MPI_COMM_WORLD)
   isum1 = mpas_globalSum(iarray5d, MPI_COMM_WORLD)
   lsum1 = mpas_globalSum(larray5d, MPI_COMM_WORLD)
   dsum2 = mpas_globalSum(darray5d, MPI_COMM_WORLD, indxRange)
   rsum2 = mpas_globalSum(rarray5d, MPI_COMM_WORLD, indxRange)
   isum2 = mpas_globalSum(iarray5d, MPI_COMM_WORLD, indxRange)
   lsum2 = mpas_globalSum(larray5d, MPI_COMM_WORLD, indxRange)
   dsum3 = mpas_globalSum(darray5d, dmask5d, MPI_COMM_WORLD)
   rsum3 = mpas_globalSum(rarray5d, rmask5d, MPI_COMM_WORLD)
   isum3 = mpas_globalSum(iarray5d, imask5d, MPI_COMM_WORLD)
   lsum3 = mpas_globalSum(larray5d, lmask5d, MPI_COMM_WORLD)
   dsum4 = mpas_globalSum(darray5d, dmask5d, MPI_COMM_WORLD, indxRange)
   rsum4 = mpas_globalSum(rarray5d, rmask5d, MPI_COMM_WORLD, indxRange)
   isum4 = mpas_globalSum(iarray5d, imask5d, MPI_COMM_WORLD, indxRange)
   lsum4 = mpas_globalSum(larray5d, lmask5d, MPI_COMM_WORLD, indxRange)
   if (dsum1 == dref) then
      print *, 'Global sum array5d double: PASS'
   else
      print *, 'Global sum array5d double: FAIL', dsum1, dref
   endif
   if (rsum1 == rref) then
      print *, 'Global sum array5d real: PASS'
   else
      print *, 'Global sum array5d real: FAIL', rsum1, rref
   endif
   if (isum1 == iref) then
      print *, 'Global sum array5d integer: PASS'
   else
      print *, 'Global sum array5d integer: FAIL', isum1, iref
   endif
   if (lsum1 == lref) then
      print *, 'Global sum array5d integer(8): PASS'
   else
      print *, 'Global sum array5d integer(8): FAIL', lsum1, lref
   endif
   if (dsum2 == dref2) then
      print *, 'Global sum irange array5d double: PASS'
   else
      print *, 'Global sum irange array5d double: FAIL', dsum2, dref2
   endif
   if (rsum2 == rref2) then
      print *, 'Global sum irange array5d real: PASS'
   else
      print *, 'Global sum irange array5d real: FAIL', rsum2, rref2
   endif
   if (isum2 == iref2) then
      print *, 'Global sum irange array5d integer: PASS'
   else
      print *, 'Global sum irange array5d integer: FAIL', isum2, iref2
   endif
   if (lsum2 == lref2) then
      print *, 'Global sum irange array5d integer(8): PASS'
   else
      print *, 'Global sum irange array5d integer(8): FAIL', lsum2, lref2
   endif
   if (dsum3 == dref2) then
      print *, 'Global sum prod array5d double: PASS'
   else
      print *, 'Global sum prod array5d double: FAIL', dsum3, dref2
   endif
   if (rsum3 == rref2) then
      print *, 'Global sum prod array5d real: PASS'
   else
      print *, 'Global sum prod array5d real: FAIL', rsum3, rref2
   endif
   if (isum3 == iref2) then
      print *, 'Global sum prod array5d integer: PASS'
   else
      print *, 'Global sum prod array5d integer: FAIL', isum3, iref2
   endif
   if (lsum3 == lref2) then
      print *, 'Global sum prod array5d integer(8): PASS'
   else
      print *, 'Global sum prod array5d integer(8): FAIL', lsum3, lref2
   endif
   if (dsum4 == dref2) then
      print *, 'Global sum prod irange array5d double: PASS'
   else
      print *, 'Global sum prod irange array5d double: FAIL', dsum4, dref2
   endif
   if (rsum4 == rref2) then
      print *, 'Global sum prod irange array5d real: PASS'
   else
      print *, 'Global sum prod irange array5d real: FAIL', rsum4, rref2
   endif
   if (isum4 == iref2) then
      print *, 'Global sum prod irange array5d integer: PASS'
   else
      print *, 'Global sum prod irange array5d integer: FAIL', isum4, iref2
   endif
   if (lsum4 == lref2) then
      print *, 'Global sum prod irange array5d integer(8): PASS'
   else
      print *, 'Global sum prod irange array5d integer(8): FAIL', lsum4, lref2
   endif
   deallocate(darray5d, dmask5d, rarray5d, rmask5d, &
              iarray5d, imask5d, larray5d, lmask5d)

   !*** 6-d arrays
   if (myRank == masterTask) print *,'Testing 6d sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray6d(nx,ny,nz,nl,nm,nn), dmask6d(nx,ny,nz,nl,nm,nn), &
            rarray6d(nx,ny,nz,nl,nm,nn), rmask6d(nx,ny,nz,nl,nm,nn), &
            iarray6d(nx,ny,nz,nl,nm,nn), imask6d(nx,ny,nz,nl,nm,nn), &
            larray6d(nx,ny,nz,nl,nm,nn), lmask6d(nx,ny,nz,nl,nm,nn))
   dref = 0.d0
   rref = 0.0
   iref = 0
   lref = 0
   dref2 = 0.d0
   rref2 = 0.0
   iref2 = 0
   lref2 = 0
   do n=1,nn
   do m=1,nm
   do l=1,nl
   do k=1,nz
   do j=1,ny
   do i=1,nx
      lscalar = (i+j+k+l+m+n)*(myRank+1)
      darray6d(i,j,k,l,m,n) = real(lscalar,R8)
      rarray6d(i,j,k,l,m,n) = real(lscalar)
      larray6d(i,j,k,l,m,n) = lscalar
      iarray6d(i,j,k,l,m,n) = min(lscalar,maxint)
      do r=0,nRanks-1
         dref = dref + real((i+j+k+l+m+n)*(r+1),R8)
         lref = lref + (i+j+k+l+m+n)*(r+1)
      end do
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4) .and. &
          k >= indxRange(5) .and. k <= indxRange(6) .and. &
          l >= indxRange(7) .and. l <= indxRange(8) .and. &
          m >= indxRange(9) .and. m <= indxRange(10) .and. &
          n >= indxRange(11) .and. n <= indxRange(12)) then
         dmask6d(i,j,k,l,m,n) = 1.d0
         rmask6d(i,j,k,l,m,n) = 1.0
         lmask6d(i,j,k,l,m,n) = 1
         imask6d(i,j,k,l,m,n) = 1
         do r=0,nRanks-1
            dref2 = dref2 + real((i+j+k+l+m+n)*(r+1),R8)
            lref2 = lref2 + (i+j+k+l+m+n)*(r+1)
         end do
      else
         dmask6d(i,j,k,l,m,n) = 0.d0
         rmask6d(i,j,k,l,m,n) = 0.0
         lmask6d(i,j,k,l,m,n) = 0
         imask6d(i,j,k,l,m,n) = 0
      endif
   end do
   end do
   end do
   end do
   end do
   end do
   rref  = dref
   rref2 = dref2
   iref  = lref
   iref2 = lref2
   dsum1 = mpas_globalSum(darray6d, MPI_COMM_WORLD)
   rsum1 = mpas_globalSum(rarray6d, MPI_COMM_WORLD)
   isum1 = mpas_globalSum(iarray6d, MPI_COMM_WORLD)
   lsum1 = mpas_globalSum(larray6d, MPI_COMM_WORLD)
   dsum2 = mpas_globalSum(darray6d, MPI_COMM_WORLD, indxRange)
   rsum2 = mpas_globalSum(rarray6d, MPI_COMM_WORLD, indxRange)
   isum2 = mpas_globalSum(iarray6d, MPI_COMM_WORLD, indxRange)
   lsum2 = mpas_globalSum(larray6d, MPI_COMM_WORLD, indxRange)
   dsum3 = mpas_globalSum(darray6d, dmask6d, MPI_COMM_WORLD)
   rsum3 = mpas_globalSum(rarray6d, rmask6d, MPI_COMM_WORLD)
   isum3 = mpas_globalSum(iarray6d, imask6d, MPI_COMM_WORLD)
   lsum3 = mpas_globalSum(larray6d, lmask6d, MPI_COMM_WORLD)
   dsum4 = mpas_globalSum(darray6d, dmask6d, MPI_COMM_WORLD, indxRange)
   rsum4 = mpas_globalSum(rarray6d, rmask6d, MPI_COMM_WORLD, indxRange)
   isum4 = mpas_globalSum(iarray6d, imask6d, MPI_COMM_WORLD, indxRange)
   lsum4 = mpas_globalSum(larray6d, lmask6d, MPI_COMM_WORLD, indxRange)
   if (dsum1 == dref) then
      print *, 'Global sum array6d double: PASS'
   else
      print *, 'Global sum array6d double: FAIL', dsum1, dref
   endif
   if (rsum1 == rref) then
      print *, 'Global sum array6d real: PASS'
   else
      print *, 'Global sum array6d real: FAIL', rsum1, rref
   endif
   if (isum1 == iref) then
      print *, 'Global sum array6d integer: PASS'
   else
      print *, 'Global sum array6d integer: FAIL', isum1, iref
   endif
   if (lsum1 == lref) then
      print *, 'Global sum array6d integer(8): PASS'
   else
      print *, 'Global sum array6d integer(8): FAIL', lsum1, lref
   endif
   if (dsum2 == dref2) then
      print *, 'Global sum irange array6d double: PASS'
   else
      print *, 'Global sum irange array6d double: FAIL', dsum2, dref2
   endif
   if (rsum2 == rref2) then
      print *, 'Global sum irange array6d real: PASS'
   else
      print *, 'Global sum irange array6d real: FAIL', rsum2, rref2
   endif
   if (isum2 == iref2) then
      print *, 'Global sum irange array6d integer: PASS'
   else
      print *, 'Global sum irange array6d integer: FAIL', isum2, iref2
   endif
   if (lsum2 == lref2) then
      print *, 'Global sum irange array6d integer(8): PASS'
   else
      print *, 'Global sum irange array6d integer(8): FAIL', lsum2, lref2
   endif
   if (dsum3 == dref2) then
      print *, 'Global sum prod array6d double: PASS'
   else
      print *, 'Global sum prod array6d double: FAIL', dsum3, dref2
   endif
   if (rsum3 == rref2) then
      print *, 'Global sum prod array6d real: PASS'
   else
      print *, 'Global sum prod array6d real: FAIL', rsum3, rref2
   endif
   if (isum3 == iref2) then
      print *, 'Global sum prod array6d integer: PASS'
   else
      print *, 'Global sum prod array6d integer: FAIL', isum3, iref2
   endif
   if (lsum3 == lref2) then
      print *, 'Global sum prod array6d integer(8): PASS'
   else
      print *, 'Global sum prod array6d integer(8): FAIL', lsum3, lref2
   endif
   if (dsum4 == dref2) then
      print *, 'Global sum prod irange array6d double: PASS'
   else
      print *, 'Global sum prod irange array6d double: FAIL', dsum4, dref2
   endif
   if (rsum4 == rref2) then
      print *, 'Global sum prod irange array6d real: PASS'
   else
      print *, 'Global sum prod irange array6d real: FAIL', rsum4, rref2
   endif
   if (isum4 == iref2) then
      print *, 'Global sum prod irange array6d integer: PASS'
   else
      print *, 'Global sum prod irange array6d integer: FAIL', isum4, iref2
   endif
   if (lsum4 == lref2) then
      print *, 'Global sum prod irange array6d integer(8): PASS'
   else
      print *, 'Global sum prod irange array6d integer(8): FAIL', lsum4, lref2
   endif
   deallocate(darray6d, dmask6d, rarray6d, rmask6d, &
              iarray6d, imask6d, larray6d, lmask6d)

   if (myRank == masterTask) print *,'Testing basic sums complete '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   !--------------------------------------------------------------------
   ! Test multi-field interfaces and check for reproducibility by
   ! using a different ordering for each of three fields.
   ! Fill the arrays with random values that span the full range of a
   ! datatype to induce roundoff errors during the sums.
   !--------------------------------------------------------------------

   !*** Scalars
   if (myRank == masterTask) print *,'Testing scalar multi-field sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(dsum1n(nFields), dsum2n(nFields), dsum3n(nFields), dsum4n(nFields), &
            rsum1n(nFields), rsum2n(nFields), rsum3n(nFields), rsum4n(nFields), &
            isum1n(nFields), isum2n(nFields), isum3n(nFields), isum4n(nFields), &
            lsum1n(nFields), lsum2n(nFields), lsum3n(nFields), lsum4n(nFields))
   allocate(darray0n(nFields), rarray0n(nFields), & 
            iarray0n(nFields), larray0n(nFields)) 
   allocate (randoms(nRanks))
   drange = 15
   rrange = 7
   irange = huge(iref)/100
   lrange = huge(lref)/100

   ! create a random number for each MPI rank and broadcast so we
   ! can check reproducibility by using the same numbers in different
   ! orders
   call random_number(randoms)
   call MPI_Bcast(randoms, nRanks, MPI_REAL, masterTask, MPI_COMM_WORLD, ierr)

   rscalar = randoms(myRank+1)
   ipower = int(rscalar*drange) - 5
   darray0n(1) = real(rscalar,R8)*(10.d0**ipower)
   ipower = int(rscalar*rrange) - 5
   rarray0n(1) = rscalar*(10.0**ipower)
   iarray0n(1) = nint(rscalar*irange)
   larray0n(1) = nint(rscalar*lrange)

   ! Second field values are in inverse order
   rscalar = randoms(nRanks-myRank)
   ipower = int(rscalar*drange) - 5
   darray0n(2) = real(rscalar,R8)*(10.d0**ipower)
   ipower = int(rscalar*rrange) - 5
   rarray0n(2) = rscalar*(10.0**ipower)
   iarray0n(2) = nint(rscalar*irange)
   larray0n(2) = nint(rscalar*lrange)

   ! Third field has values another order
   iref = mod(nRanks/2+myRank ,nRanks) + 1
   rscalar = randoms(iref)
   ipower = int(rscalar*drange) - 5
   darray0n(3) = real(rscalar,R8)*(10.d0**ipower)
   ipower = int(rscalar*rrange) - 5
   rarray0n(3) = rscalar*(10.0**ipower)
   iarray0n(3) = nint(rscalar*irange)
   larray0n(3) = nint(rscalar*lrange)

   dsum1n(:) = 0.d0
   rsum1n(:) = 0.0
   isum1n(:) = 0
   lsum1n(:) = 0
   do r=0,nRanks-1
      iref = r+1
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      dsum1n(1) = dsum1n(1) + real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rsum1n(1) = rsum1n(1) + rscalar*(10.0**ipower)
      isum1n(1) = isum1n(1) + nint(rscalar*irange)
      lsum1n(1) = lsum1n(1) + nint(rscalar*lrange)

      iref = nRanks - r
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      dsum1n(2) = dsum1n(2) + real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rsum1n(2) = rsum1n(2) + rscalar*(10.0**ipower)
      isum1n(2) = isum1n(2) + nint(rscalar*irange)
      lsum1n(2) = lsum1n(2) + nint(rscalar*lrange)

      iref = mod(nRanks/2+r ,nRanks) + 1
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      dsum1n(3) = dsum1n(3) + real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rsum1n(3) = rsum1n(3) + rscalar*(10.0**ipower)
      isum1n(3) = isum1n(3) + nint(rscalar*irange)
      lsum1n(3) = lsum1n(3) + nint(rscalar*lrange)

   end do
   ! Compute a reference sum using the single field calls from above
   dscalar = darray0n(1)
   rscalar = rarray0n(1)
   iscalar = iarray0n(1)
   lscalar = larray0n(1)
   dref = mpas_globalSum(dscalar, MPI_COMM_WORLD)
   rref = mpas_globalSum(rscalar, MPI_COMM_WORLD)
   iref = mpas_globalSum(iscalar, MPI_COMM_WORLD)
   lref = mpas_globalSum(lscalar, MPI_COMM_WORLD)

   ! Check that the reference sums for floats and doubles
   ! are dependent on order of summation and then ints/longs are not
   if (lsum1n(1) /= lref .or. lsum1n(2) /= lref .or. &
       lsum1n(3) /= lref) then
      print *, 'Multi-field setup scalar integer(8): FAIL', lref, lsum1n
   endif
   if (isum1n(1) /= iref .or. isum1n(2) /= iref .or. &
       isum1n(3) /= iref) then
      print *, 'Multi-field setup scalar integer: FAIL', iref, isum1n
   endif
   if (dsum1n(1) == dsum1n(2) .and. dsum1n(1) == dsum1n(3) .and. &
       dsum1n(2) == dsum1n(3)) then
       print *, 'Multi-field double scalar not sensitive to roundoff : WARN', dref, dsum1n
       print *, 'Multi-field double scalar data:', myRank, darray0n
   endif
   if (rsum1n(1) == rsum1n(2) .and. rsum1n(1) == rsum1n(3) .and. &
       rsum1n(2) == rsum1n(3)) then
       print *, 'Multi-field real scalar not sensitive to roundoff : WARN', rref, rsum1n
       print *, 'Multi-field real scalar data:', myRank, rarray0n
   endif

   ! Now do actual multi-field and reproducibility tests
   dsum1n = mpas_globalSumNfld(darray0n, MPI_COMM_WORLD)
   rsum1n = mpas_globalSumNfld(rarray0n, MPI_COMM_WORLD)
   isum1n = mpas_globalSumNfld(iarray0n, MPI_COMM_WORLD)
   lsum1n = mpas_globalSumNfld(larray0n, MPI_COMM_WORLD)
   if (dsum1n(1) == dref .and. dsum1n(2) == dref .and. &
       dsum1n(3) == dref) then
      print *, 'Reproducibility and nfld scalar double: PASS'
   else
      print *, 'Reproducibility and nfld scalar double: FAIL', dref, dsum1n
   endif
   if (rsum1n(1) == rref .and. rsum1n(2) == rref .and. &
       rsum1n(3) == rref) then
      print *, 'Reproducibility and nfld scalar real: PASS'
   else
      print *, 'Reproducibility and nfld scalar real: FAIL', rref, rsum1n
   endif
   if (isum1n(1) == iref .and. isum1n(2) == iref .and. &
       isum1n(3) == iref) then
      print *, 'Reproducibility and nfld scalar integer: PASS'
   else
      print *, 'Reproducibility and nfld scalar integer: FAIL', iref, isum1n
   endif
   if (lsum1n(1) == lref .and. lsum1n(2) == lref .and. &
       lsum1n(3) == lref) then
      print *, 'Reproducibility and nfld scalar integer(8): PASS'
   else
      print *, 'Reproducibility and nfld scalar integer(8): FAIL', lref, lsum1n
   endif
   deallocate(darray0n, rarray0n, iarray0n, larray0n)
   deallocate(randoms)

   !*** 1-d arrays
   if (myRank == masterTask) print *,'Testing 1d multi-field sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray1n(nx,nFields), dmask1n(nx,nFields), &
            rarray1n(nx,nFields), rmask1n(nx,nFields), &
            iarray1n(nx,nFields), imask1n(nx,nFields), &
            larray1n(nx,nFields), lmask1n(nx,nFields))
   allocate(randoms(nx))
   call random_number(randoms)

   iref = 0
   do i=1,nx
      iref = iref + 1
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      darray1n(i,1) = real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rarray1n(i,1) = rscalar*(10.0**ipower)
      iarray1n(i,1) = nint(rscalar*irange)
      larray1n(i,1) = nint(rscalar*lrange)
   end do
   do i=1,nx
      i2 = nx+1-i
      i3 = mod(nx/2+i,nx) + 1
      darray1n(i,2) = darray1n(i2,1)
      darray1n(i,3) = darray1n(i3,1)
      rarray1n(i,2) = rarray1n(i2,1)
      rarray1n(i,3) = rarray1n(i3,1)
      iarray1n(i,2) = iarray1n(i2,1)
      iarray1n(i,3) = iarray1n(i3,1)
      larray1n(i,2) = larray1n(i2,1)
      larray1n(i,3) = larray1n(i3,1)
   end do

   do nfld=1,nFields
   do i=1,nx
      if (i >= indxRange(1) .and. i <= indxRange(2)) then
         dmask1n(i,nfld) = 1.d0
         rmask1n(i,nfld) = 1.0
         lmask1n(i,nfld) = 1
         imask1n(i,nfld) = 1
      else
         dmask1n(i,nfld) = 0.d0
         rmask1n(i,nfld) = 0.0
         lmask1n(i,nfld) = 0
         imask1n(i,nfld) = 0
      endif
   end do
   end do
   ! Use the single-field interface to compute reference sums
   dref = mpas_globalSum(darray1n(:,1), MPI_COMM_WORLD)
   rref = mpas_globalSum(rarray1n(:,1), MPI_COMM_WORLD)
   iref = mpas_globalSum(iarray1n(:,1), MPI_COMM_WORLD)
   lref = mpas_globalSum(larray1n(:,1), MPI_COMM_WORLD)

   ! Now compute the various multi-field sums
   dsum1n = mpas_globalSumNfld(darray1n, MPI_COMM_WORLD)
   rsum1n = mpas_globalSumNfld(rarray1n, MPI_COMM_WORLD)
   isum1n = mpas_globalSumNfld(iarray1n, MPI_COMM_WORLD)
   lsum1n = mpas_globalSumNfld(larray1n, MPI_COMM_WORLD)
   dsum2n = mpas_globalSumNfld(darray1n, MPI_COMM_WORLD, indxRange)
   rsum2n = mpas_globalSumNfld(rarray1n, MPI_COMM_WORLD, indxRange)
   isum2n = mpas_globalSumNfld(iarray1n, MPI_COMM_WORLD, indxRange)
   lsum2n = mpas_globalSumNfld(larray1n, MPI_COMM_WORLD, indxRange)
   dsum3n = mpas_globalSumNfld(darray1n, dmask1n, MPI_COMM_WORLD)
   rsum3n = mpas_globalSumNfld(rarray1n, rmask1n, MPI_COMM_WORLD)
   isum3n = mpas_globalSumNfld(iarray1n, imask1n, MPI_COMM_WORLD)
   lsum3n = mpas_globalSumNfld(larray1n, lmask1n, MPI_COMM_WORLD)
   dsum4n = mpas_globalSumNfld(darray1n, dmask1n, MPI_COMM_WORLD, indxRange)
   rsum4n = mpas_globalSumNfld(rarray1n, rmask1n, MPI_COMM_WORLD, indxRange)
   isum4n = mpas_globalSumNfld(iarray1n, imask1n, MPI_COMM_WORLD, indxRange)
   lsum4n = mpas_globalSumNfld(larray1n, lmask1n, MPI_COMM_WORLD, indxRange)

   if (dsum1n(1) == dref .and. dsum1n(2) == dref .and. &
       dsum1n(3) == dref) then
      print *, 'Multi-field and reproducibility 1d double: PASS'
   else
      print *, 'Multi-field and reproducibility 1d double: FAIL', dref, dsum1n
   endif
   if (rsum1n(1) == rref .and. rsum1n(2) == rref .and. &
       rsum1n(3) == rref) then
      print *, 'Multi-field and reproducibility 1d real: PASS'
   else
      print *, 'Multi-field and reproducibility 1d real: FAIL', rref, rsum1n
   endif
   if (isum1n(1) == iref .and. isum1n(2) == iref .and. &
       isum1n(3) == iref) then
      print *, 'Multi-field and reproducibility 1d int: PASS'
   else
      print *, 'Multi-field and reproducibility 1d int: FAIL', iref, isum1n
   endif
   if (lsum1n(1) == lref .and. lsum1n(2) == lref .and. &
       lsum1n(3) == lref) then
      print *, 'Multi-field and reproducibility 1d int(8): PASS'
   else
      print *, 'Multi-field and reproducibility 1d int(8): FAIL', lref, lsum1n
   endif
   isum1 = 0
   isum2 = 0
   isum3 = 0
   isum4 = 0
   do nfld=1,nFields
      if (dsum2n(nfld) /= dsum3n(nfld)) isum1 = isum1+1
      if (dsum2n(nfld) /= dsum4n(nfld)) isum1 = isum1+1
      if (rsum2n(nfld) /= rsum3n(nfld)) isum2 = isum2+1
      if (rsum2n(nfld) /= rsum4n(nfld)) isum2 = isum2+1
      if (isum2n(nfld) /= isum3n(nfld)) isum3 = isum3+1
      if (isum2n(nfld) /= isum4n(nfld)) isum3 = isum3+1
      if (lsum2n(nfld) /= lsum3n(nfld)) isum4 = isum4+1
      if (lsum2n(nfld) /= lsum4n(nfld)) isum4 = isum4+1
   end do
   if (isum1 > 0) then
      print *, 'Multifield range/mask 1d double: FAIL', dsum2n,dsum3n,dsum4n
   else
      print *, 'Multifield range/mask 1d double: PASS'
   endif
   if (isum2 > 0) then
      print *, 'Multifield range/mask 1d real: FAIL', rsum2n,rsum3n,rsum4n
   else
      print *, 'Multifield range/mask 1d real: PASS'
   endif
   if (isum3 > 0) then
      print *, 'Multifield range/mask 1d int: FAIL', isum2n,isum3n,isum4n
   else
      print *, 'Multifield range/mask 1d int: PASS'
   endif
   if (isum4 > 0) then
      print *, 'Multifield range/mask 1d int(8): FAIL', lsum2n,lsum3n,lsum4n
   else
      print *, 'Multifield range/mask 1d int(8): PASS'
   endif
   deallocate(darray1n, dmask1n, rarray1n, rmask1n, &
              iarray1n, imask1n, larray1n, lmask1n)
   deallocate(randoms)

   !*** 2-d arrays
   if (myRank == masterTask) print *,'Testing 2d multi-field sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray2n(nx,ny,nFields), dmask2n(nx,ny,nFields), &
            rarray2n(nx,ny,nFields), rmask2n(nx,ny,nFields), &
            iarray2n(nx,ny,nFields), imask2n(nx,ny,nFields), &
            larray2n(nx,ny,nFields), lmask2n(nx,ny,nFields))
   allocate(randoms(nx*ny))
   call random_number(randoms)

   iref = 0
   do j=1,ny
   do i=1,nx
      iref = iref + 1
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      darray2n(i,j,1) = real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rarray2n(i,j,1) = rscalar*(10.0**ipower)
      iarray2n(i,j,1) = nint(rscalar*irange)
      larray2n(i,j,1) = nint(rscalar*lrange)
   end do
   end do
   do j=1,ny
      j2 = ny+1-j
      j3 = mod(ny/2+j,ny) + 1
      do i=1,nx
         i2 = nx+1-i
         i3 = mod(nx/2+i,nx) + 1
         darray2n(i,j,2) = darray2n(i2,j2,1)
         darray2n(i,j,3) = darray2n(i3,j3,1)
         rarray2n(i,j,2) = rarray2n(i2,j2,1)
         rarray2n(i,j,3) = rarray2n(i3,j3,1)
         iarray2n(i,j,2) = iarray2n(i2,j2,1)
         iarray2n(i,j,3) = iarray2n(i3,j3,1)
         larray2n(i,j,2) = larray2n(i2,j2,1)
         larray2n(i,j,3) = larray2n(i3,j3,1)
      end do
   end do

   do nfld=1,nFields
   do j=1,ny
   do i=1,nx
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4)) then
         dmask2n(i,j,nfld) = 1.d0
         rmask2n(i,j,nfld) = 1.0
         lmask2n(i,j,nfld) = 1
         imask2n(i,j,nfld) = 1
      else
         dmask2n(i,j,nfld) = 0.d0
         rmask2n(i,j,nfld) = 0.0
         lmask2n(i,j,nfld) = 0
         imask2n(i,j,nfld) = 0
      endif
   end do
   end do
   end do
   ! Use the single-field interface to compute reference sums
   dref = mpas_globalSum(darray2n(:,:,1), MPI_COMM_WORLD)
   rref = mpas_globalSum(rarray2n(:,:,1), MPI_COMM_WORLD)
   iref = mpas_globalSum(iarray2n(:,:,1), MPI_COMM_WORLD)
   lref = mpas_globalSum(larray2n(:,:,1), MPI_COMM_WORLD)

   ! Now compute the various multi-field sums
   dsum1n = mpas_globalSumNfld(darray2n, MPI_COMM_WORLD)
   rsum1n = mpas_globalSumNfld(rarray2n, MPI_COMM_WORLD)
   isum1n = mpas_globalSumNfld(iarray2n, MPI_COMM_WORLD)
   lsum1n = mpas_globalSumNfld(larray2n, MPI_COMM_WORLD)
   dsum2n = mpas_globalSumNfld(darray2n, MPI_COMM_WORLD, indxRange)
   rsum2n = mpas_globalSumNfld(rarray2n, MPI_COMM_WORLD, indxRange)
   isum2n = mpas_globalSumNfld(iarray2n, MPI_COMM_WORLD, indxRange)
   lsum2n = mpas_globalSumNfld(larray2n, MPI_COMM_WORLD, indxRange)
   dsum3n = mpas_globalSumNfld(darray2n, dmask2n, MPI_COMM_WORLD)
   rsum3n = mpas_globalSumNfld(rarray2n, rmask2n, MPI_COMM_WORLD)
   isum3n = mpas_globalSumNfld(iarray2n, imask2n, MPI_COMM_WORLD)
   lsum3n = mpas_globalSumNfld(larray2n, lmask2n, MPI_COMM_WORLD)
   dsum4n = mpas_globalSumNfld(darray2n, dmask2n, MPI_COMM_WORLD, indxRange)
   rsum4n = mpas_globalSumNfld(rarray2n, rmask2n, MPI_COMM_WORLD, indxRange)
   isum4n = mpas_globalSumNfld(iarray2n, imask2n, MPI_COMM_WORLD, indxRange)
   lsum4n = mpas_globalSumNfld(larray2n, lmask2n, MPI_COMM_WORLD, indxRange)

   if (dsum1n(1) == dref .and. dsum1n(2) == dref .and. &
       dsum1n(3) == dref) then
      print *, 'Multi-field and reproducibility 2d double: PASS'
   else
      print *, 'Multi-field and reproducibility 2d double: FAIL', dref, dsum1n
   endif
   if (rsum1n(1) == rref .and. rsum1n(2) == rref .and. &
       rsum1n(3) == rref) then
      print *, 'Multi-field and reproducibility 2d real: PASS'
   else
      print *, 'Multi-field and reproducibility 2d real: FAIL', rref, rsum1n
   endif
   if (isum1n(1) == iref .and. isum1n(2) == iref .and. &
       isum1n(3) == iref) then
      print *, 'Multi-field and reproducibility 2d int: PASS'
   else
      print *, 'Multi-field and reproducibility 2d int: FAIL', iref, isum1n
   endif
   if (lsum1n(1) == lref .and. lsum1n(2) == lref .and. &
       lsum1n(3) == lref) then
      print *, 'Multi-field and reproducibility 2d int(8): PASS'
   else
      print *, 'Multi-field and reproducibility 2d int(8): FAIL', lref, lsum1n
   endif
   isum1 = 0
   isum2 = 0
   isum3 = 0
   isum4 = 0
   do nfld=1,nFields
      if (dsum2n(nfld) /= dsum3n(nfld)) isum1 = isum1+1
      if (dsum2n(nfld) /= dsum4n(nfld)) isum1 = isum1+1
      if (rsum2n(nfld) /= rsum3n(nfld)) isum2 = isum2+1
      if (rsum2n(nfld) /= rsum4n(nfld)) isum2 = isum2+1
      if (isum2n(nfld) /= isum3n(nfld)) isum3 = isum3+1
      if (isum2n(nfld) /= isum4n(nfld)) isum3 = isum3+1
      if (lsum2n(nfld) /= lsum3n(nfld)) isum4 = isum4+1
      if (lsum2n(nfld) /= lsum4n(nfld)) isum4 = isum4+1
   end do
   if (isum1 > 0) then
      print *, 'Multifield range/mask 2d double: FAIL', dsum2n,dsum3n,dsum4n
   else
      print *, 'Multifield range/mask 2d double: PASS'
   endif
   if (isum2 > 0) then
      print *, 'Multifield range/mask 2d real: FAIL', rsum2n,rsum3n,rsum4n
   else
      print *, 'Multifield range/mask 2d real: PASS'
   endif
   if (isum3 > 0) then
      print *, 'Multifield range/mask 2d int: FAIL', isum2n,isum3n,isum4n
   else
      print *, 'Multifield range/mask 2d int: PASS'
   endif
   if (isum4 > 0) then
      print *, 'Multifield range/mask 2d int(8): FAIL', lsum2n,lsum3n,lsum4n
   else
      print *, 'Multifield range/mask 2d int(8): PASS'
   endif
   deallocate(darray2n, dmask2n, rarray2n, rmask2n, &
              iarray2n, imask2n, larray2n, lmask2n)
   deallocate(randoms)

   !*** 3-d arrays
   if (myRank == masterTask) print *,'Testing 3d multi-field sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray3n(nx,ny,nz,nFields), dmask3n(nx,ny,nz,nFields), &
            rarray3n(nx,ny,nz,nFields), rmask3n(nx,ny,nz,nFields), &
            iarray3n(nx,ny,nz,nFields), imask3n(nx,ny,nz,nFields), &
            larray3n(nx,ny,nz,nFields), lmask3n(nx,ny,nz,nFields))
   allocate(randoms(nx*ny*nz))
   call random_number(randoms)

   iref = 0
   do k=1,nz
   do j=1,ny
   do i=1,nx
      iref = iref + 1
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      darray3n(i,j,k,1) = real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rarray3n(i,j,k,1) = rscalar*(10.0**ipower)
      iarray3n(i,j,k,1) = nint(rscalar*irange)
      larray3n(i,j,k,1) = nint(rscalar*lrange)
   end do
   end do
   end do
   do k=1,nz
      k2 = nz+1-k
      k3 = mod(nz/2+k,nz) + 1
      do j=1,ny
         j2 = ny+1-j
         j3 = mod(ny/2+j,ny) + 1
         do i=1,nx
            i2 = nx+1-i
            i3 = mod(nx/2+i,nx) + 1
            darray3n(i,j,k,2) = darray3n(i2,j2,k2,1)
            darray3n(i,j,k,3) = darray3n(i3,j3,k3,1)
            rarray3n(i,j,k,2) = rarray3n(i2,j2,k2,1)
            rarray3n(i,j,k,3) = rarray3n(i3,j3,k3,1)
            iarray3n(i,j,k,2) = iarray3n(i2,j2,k2,1)
            iarray3n(i,j,k,3) = iarray3n(i3,j3,k3,1)
            larray3n(i,j,k,2) = larray3n(i2,j2,k2,1)
            larray3n(i,j,k,3) = larray3n(i3,j3,k3,1)
         end do
      end do
   end do

   do nfld=1,nFields
   do k=1,nz
   do j=1,ny
   do i=1,nx
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4) .and. &
          k >= indxRange(5) .and. k <= indxRange(6)) then
         dmask3n(i,j,k,nfld) = 1.d0
         rmask3n(i,j,k,nfld) = 1.0
         lmask3n(i,j,k,nfld) = 1
         imask3n(i,j,k,nfld) = 1
      else
         dmask3n(i,j,k,nfld) = 0.d0
         rmask3n(i,j,k,nfld) = 0.0
         lmask3n(i,j,k,nfld) = 0
         imask3n(i,j,k,nfld) = 0
      endif
   end do
   end do
   end do
   end do
   ! Use the single-field interface to compute reference sums
   dref = mpas_globalSum(darray3n(:,:,:,1), MPI_COMM_WORLD)
   rref = mpas_globalSum(rarray3n(:,:,:,1), MPI_COMM_WORLD)
   iref = mpas_globalSum(iarray3n(:,:,:,1), MPI_COMM_WORLD)
   lref = mpas_globalSum(larray3n(:,:,:,1), MPI_COMM_WORLD)

   ! Now compute the various multi-field sums
   dsum1n = mpas_globalSumNfld(darray3n, MPI_COMM_WORLD)
   rsum1n = mpas_globalSumNfld(rarray3n, MPI_COMM_WORLD)
   isum1n = mpas_globalSumNfld(iarray3n, MPI_COMM_WORLD)
   lsum1n = mpas_globalSumNfld(larray3n, MPI_COMM_WORLD)
   dsum2n = mpas_globalSumNfld(darray3n, MPI_COMM_WORLD, indxRange)
   rsum2n = mpas_globalSumNfld(rarray3n, MPI_COMM_WORLD, indxRange)
   isum2n = mpas_globalSumNfld(iarray3n, MPI_COMM_WORLD, indxRange)
   lsum2n = mpas_globalSumNfld(larray3n, MPI_COMM_WORLD, indxRange)
   dsum3n = mpas_globalSumNfld(darray3n, dmask3n, MPI_COMM_WORLD)
   rsum3n = mpas_globalSumNfld(rarray3n, rmask3n, MPI_COMM_WORLD)
   isum3n = mpas_globalSumNfld(iarray3n, imask3n, MPI_COMM_WORLD)
   lsum3n = mpas_globalSumNfld(larray3n, lmask3n, MPI_COMM_WORLD)
   dsum4n = mpas_globalSumNfld(darray3n, dmask3n, MPI_COMM_WORLD, indxRange)
   rsum4n = mpas_globalSumNfld(rarray3n, rmask3n, MPI_COMM_WORLD, indxRange)
   isum4n = mpas_globalSumNfld(iarray3n, imask3n, MPI_COMM_WORLD, indxRange)
   lsum4n = mpas_globalSumNfld(larray3n, lmask3n, MPI_COMM_WORLD, indxRange)

   if (dsum1n(1) == dref .and. dsum1n(2) == dref .and. &
       dsum1n(3) == dref) then
      print *, 'Multi-field and reproducibility 3d double: PASS'
   else
      print *, 'Multi-field and reproducibility 3d double: FAIL', dref, dsum1n
   endif
   if (rsum1n(1) == rref .and. rsum1n(2) == rref .and. &
       rsum1n(3) == rref) then
      print *, 'Multi-field and reproducibility 3d real: PASS'
   else
      print *, 'Multi-field and reproducibility 3d real: FAIL', rref, rsum1n
   endif
   if (isum1n(1) == iref .and. isum1n(2) == iref .and. &
       isum1n(3) == iref) then
      print *, 'Multi-field and reproducibility 3d int: PASS'
   else
      print *, 'Multi-field and reproducibility 3d int: FAIL', iref, isum1n
   endif
   if (lsum1n(1) == lref .and. lsum1n(2) == lref .and. &
       lsum1n(3) == lref) then
      print *, 'Multi-field and reproducibility 3d int(8): PASS'
   else
      print *, 'Multi-field and reproducibility 3d int(8): FAIL', lref, lsum1n
   endif
   isum1 = 0
   isum2 = 0
   isum3 = 0
   isum4 = 0
   do nfld=1,nFields
      if (dsum2n(nfld) /= dsum3n(nfld)) isum1 = isum1+1
      if (dsum2n(nfld) /= dsum4n(nfld)) isum1 = isum1+1
      if (rsum2n(nfld) /= rsum3n(nfld)) isum2 = isum2+1
      if (rsum2n(nfld) /= rsum4n(nfld)) isum2 = isum2+1
      if (isum2n(nfld) /= isum3n(nfld)) isum3 = isum3+1
      if (isum2n(nfld) /= isum4n(nfld)) isum3 = isum3+1
      if (lsum2n(nfld) /= lsum3n(nfld)) isum4 = isum4+1
      if (lsum2n(nfld) /= lsum4n(nfld)) isum4 = isum4+1
   end do
   if (isum1 > 0) then
      print *, 'Multifield range/mask 3d double: FAIL', dsum2n,dsum3n,dsum4n
   else
      print *, 'Multifield range/mask 3d double: PASS'
   endif
   if (isum2 > 0) then
      print *, 'Multifield range/mask 3d real: FAIL', rsum2n,rsum3n,rsum4n
   else
      print *, 'Multifield range/mask 3d real: PASS'
   endif
   if (isum3 > 0) then
      print *, 'Multifield range/mask 3d int: FAIL', isum2n,isum3n,isum4n
   else
      print *, 'Multifield range/mask 3d int: PASS'
   endif
   if (isum4 > 0) then
      print *, 'Multifield range/mask 3d int(8): FAIL', lsum2n,lsum3n,lsum4n
   else
      print *, 'Multifield range/mask 3d int(8): PASS'
   endif
   deallocate(darray3n, dmask3n, rarray3n, rmask3n, &
              iarray3n, imask3n, larray3n, lmask3n)
   deallocate(randoms)

   !*** 4-d arrays
   if (myRank == masterTask) print *,'Testing 4d multi-field sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray4n(nx,ny,nz,nl,nFields), &
            rarray4n(nx,ny,nz,nl,nFields), &
            iarray4n(nx,ny,nz,nl,nFields), &
            larray4n(nx,ny,nz,nl,nFields), &
             dmask4n(nx,ny,nz,nl,nFields), &
             rmask4n(nx,ny,nz,nl,nFields), &
             imask4n(nx,ny,nz,nl,nFields), &
             lmask4n(nx,ny,nz,nl,nFields))
   allocate(randoms(nx*ny*nz*nl))
   call random_number(randoms)

   iref = 0
   do l=1,nl
   do k=1,nz
   do j=1,ny
   do i=1,nx
      iref = iref + 1
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      darray4n(i,j,k,l,1) = real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rarray4n(i,j,k,l,1) = rscalar*(10.0**ipower)
      iarray4n(i,j,k,l,1) = nint(rscalar*irange)
      larray4n(i,j,k,l,1) = nint(rscalar*lrange)
   end do
   end do
   end do
   end do
   do l=1,nl
      l2 = nl+1-l
      l3 = mod(nl/2+l,nl) + 1
      do k=1,nz
         k2 = nz+1-k
         k3 = mod(nz/2+k,nz) + 1
         do j=1,ny
            j2 = ny+1-j
            j3 = mod(ny/2+j,ny) + 1
            do i=1,nx
               i2 = nx+1-i
               i3 = mod(nx/2+i,nx) + 1
               darray4n(i,j,k,l,2) = darray4n(i2,j2,k2,l2,1)
               darray4n(i,j,k,l,3) = darray4n(i3,j3,k3,l3,1)
               rarray4n(i,j,k,l,2) = rarray4n(i2,j2,k2,l2,1)
               rarray4n(i,j,k,l,3) = rarray4n(i3,j3,k3,l3,1)
               iarray4n(i,j,k,l,2) = iarray4n(i2,j2,k2,l2,1)
               iarray4n(i,j,k,l,3) = iarray4n(i3,j3,k3,l3,1)
               larray4n(i,j,k,l,2) = larray4n(i2,j2,k2,l2,1)
               larray4n(i,j,k,l,3) = larray4n(i3,j3,k3,l3,1)
            end do
         end do
      end do
   end do

   do nfld=1,nFields
   do l=1,nl
   do k=1,nz
   do j=1,ny
   do i=1,nx
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4) .and. &
          k >= indxRange(5) .and. k <= indxRange(6) .and. &
          l >= indxRange(7) .and. l <= indxRange(8)) then
         dmask4n(i,j,k,l,nfld) = 1.d0
         rmask4n(i,j,k,l,nfld) = 1.0
         lmask4n(i,j,k,l,nfld) = 1
         imask4n(i,j,k,l,nfld) = 1
      else
         dmask4n(i,j,k,l,nfld) = 0.d0
         rmask4n(i,j,k,l,nfld) = 0.0
         lmask4n(i,j,k,l,nfld) = 0
         imask4n(i,j,k,l,nfld) = 0
      endif
   end do
   end do
   end do
   end do
   end do
   ! Use the single-field interface to compute reference sums
   dref = mpas_globalSum(darray4n(:,:,:,:,1), MPI_COMM_WORLD)
   rref = mpas_globalSum(rarray4n(:,:,:,:,1), MPI_COMM_WORLD)
   iref = mpas_globalSum(iarray4n(:,:,:,:,1), MPI_COMM_WORLD)
   lref = mpas_globalSum(larray4n(:,:,:,:,1), MPI_COMM_WORLD)

   ! Now compute the various multi-field sums
   dsum1n = mpas_globalSumNfld(darray4n, MPI_COMM_WORLD)
   rsum1n = mpas_globalSumNfld(rarray4n, MPI_COMM_WORLD)
   isum1n = mpas_globalSumNfld(iarray4n, MPI_COMM_WORLD)
   lsum1n = mpas_globalSumNfld(larray4n, MPI_COMM_WORLD)
   dsum2n = mpas_globalSumNfld(darray4n, MPI_COMM_WORLD, indxRange)
   rsum2n = mpas_globalSumNfld(rarray4n, MPI_COMM_WORLD, indxRange)
   isum2n = mpas_globalSumNfld(iarray4n, MPI_COMM_WORLD, indxRange)
   lsum2n = mpas_globalSumNfld(larray4n, MPI_COMM_WORLD, indxRange)
   dsum3n = mpas_globalSumNfld(darray4n, dmask4n, MPI_COMM_WORLD)
   rsum3n = mpas_globalSumNfld(rarray4n, rmask4n, MPI_COMM_WORLD)
   isum3n = mpas_globalSumNfld(iarray4n, imask4n, MPI_COMM_WORLD)
   lsum3n = mpas_globalSumNfld(larray4n, lmask4n, MPI_COMM_WORLD)
   dsum4n = mpas_globalSumNfld(darray4n, dmask4n, MPI_COMM_WORLD, indxRange)
   rsum4n = mpas_globalSumNfld(rarray4n, rmask4n, MPI_COMM_WORLD, indxRange)
   isum4n = mpas_globalSumNfld(iarray4n, imask4n, MPI_COMM_WORLD, indxRange)
   lsum4n = mpas_globalSumNfld(larray4n, lmask4n, MPI_COMM_WORLD, indxRange)

   if (dsum1n(1) == dref .and. dsum1n(2) == dref .and. &
       dsum1n(3) == dref) then
      print *, 'Multi-field and reproducibility 4d double: PASS'
   else
      print *, 'Multi-field and reproducibility 4d double: FAIL', dref, dsum1n
   endif
   if (rsum1n(1) == rref .and. rsum1n(2) == rref .and. &
       rsum1n(3) == rref) then
      print *, 'Multi-field and reproducibility 4d real: PASS'
   else
      print *, 'Multi-field and reproducibility 4d real: FAIL', rref, rsum1n
   endif
   if (isum1n(1) == iref .and. isum1n(2) == iref .and. &
       isum1n(3) == iref) then
      print *, 'Multi-field and reproducibility 4d int: PASS'
   else
      print *, 'Multi-field and reproducibility 4d int: FAIL', iref, isum1n
   endif
   if (lsum1n(1) == lref .and. lsum1n(2) == lref .and. &
       lsum1n(3) == lref) then
      print *, 'Multi-field and reproducibility 4d int(8): PASS'
   else
      print *, 'Multi-field and reproducibility 4d int(8): FAIL', lref, lsum1n
   endif
   isum1 = 0
   isum2 = 0
   isum3 = 0
   isum4 = 0
   do nfld=1,nFields
      if (dsum2n(nfld) /= dsum3n(nfld)) isum1 = isum1+1
      if (dsum2n(nfld) /= dsum4n(nfld)) isum1 = isum1+1
      if (rsum2n(nfld) /= rsum3n(nfld)) isum2 = isum2+1
      if (rsum2n(nfld) /= rsum4n(nfld)) isum2 = isum2+1
      if (isum2n(nfld) /= isum3n(nfld)) isum3 = isum3+1
      if (isum2n(nfld) /= isum4n(nfld)) isum3 = isum3+1
      if (lsum2n(nfld) /= lsum3n(nfld)) isum4 = isum4+1
      if (lsum2n(nfld) /= lsum4n(nfld)) isum4 = isum4+1
   end do
   if (isum1 > 0) then
      print *, 'Multifield range/mask 4d double: FAIL', dsum2n,dsum3n,dsum4n
   else
      print *, 'Multifield range/mask 4d double: PASS'
   endif
   if (isum2 > 0) then
      print *, 'Multifield range/mask 4d real: FAIL', rsum2n,rsum3n,rsum4n
   else
      print *, 'Multifield range/mask 4d real: PASS'
   endif
   if (isum3 > 0) then
      print *, 'Multifield range/mask 4d int: FAIL', isum2n,isum3n,isum4n
   else
      print *, 'Multifield range/mask 4d int: PASS'
   endif
   if (isum4 > 0) then
      print *, 'Multifield range/mask 4d int(8): FAIL', lsum2n,lsum3n,lsum4n
   else
      print *, 'Multifield range/mask 4d int(8): PASS'
   endif
   deallocate(darray4n, dmask4n, rarray4n, rmask4n, &
              iarray4n, imask4n, larray4n, lmask4n)
   deallocate(randoms)

   !*** 5-d arrays
   if (myRank == masterTask) print *,'Testing 5d multi-field sums '
   flush(6)
   call MPI_Barrier(MPI_COMM_WORLD, ierr)
   allocate(darray5n(nx,ny,nz,nl,nm,nFields), &
            rarray5n(nx,ny,nz,nl,nm,nFields), &
            iarray5n(nx,ny,nz,nl,nm,nFields), &
            larray5n(nx,ny,nz,nl,nm,nFields), &
             dmask5n(nx,ny,nz,nl,nm,nFields), &
             rmask5n(nx,ny,nz,nl,nm,nFields), &
             imask5n(nx,ny,nz,nl,nm,nFields), &
             lmask5n(nx,ny,nz,nl,nm,nFields))
   allocate(randoms(nx*ny*nz*nl*nm))
   call random_number(randoms)

   iref = 0
   do m=1,nm
   do l=1,nl
   do k=1,nz
   do j=1,ny
   do i=1,nx
      iref = iref + 1
      rscalar = randoms(iref)
      ipower = int(rscalar*drange) - 5
      darray5n(i,j,k,l,m,1) = real(rscalar,R8)*(10.d0**ipower)
      ipower = int(rscalar*rrange) - 5
      rarray5n(i,j,k,l,m,1) = rscalar*(10.0**ipower)
      iarray5n(i,j,k,l,m,1) = nint(rscalar*irange)
      larray5n(i,j,k,l,m,1) = nint(rscalar*lrange)
   end do
   end do
   end do
   end do
   end do
   do m=1,nm
      m2 = nm+1-m
      m3 = mod(nm/2+m,nm) + 1
      do l=1,nl
         l2 = nl+1-l
         l3 = mod(nl/2+l,nl) + 1
         do k=1,nz
            k2 = nz+1-k
            k3 = mod(nz/2+k,nz) + 1
            do j=1,ny
               j2 = ny+1-j
               j3 = mod(ny/2+j,ny) + 1
               do i=1,nx
                  i2 = nx+1-i
                  i3 = mod(nx/2+i,nx) + 1
                  darray5n(i,j,k,l,m,2) = darray5n(i2,j2,k2,l2,m2,1)
                  darray5n(i,j,k,l,m,3) = darray5n(i3,j3,k3,l3,m3,1)
                  rarray5n(i,j,k,l,m,2) = rarray5n(i2,j2,k2,l2,m2,1)
                  rarray5n(i,j,k,l,m,3) = rarray5n(i3,j3,k3,l3,m3,1)
                  iarray5n(i,j,k,l,m,2) = iarray5n(i2,j2,k2,l2,m2,1)
                  iarray5n(i,j,k,l,m,3) = iarray5n(i3,j3,k3,l3,m3,1)
                  larray5n(i,j,k,l,m,2) = larray5n(i2,j2,k2,l2,m2,1)
                  larray5n(i,j,k,l,m,3) = larray5n(i3,j3,k3,l3,m3,1)
               end do
            end do
         end do
      end do
   end do

   do nfld=1,nFields
   do m=1,nm
   do l=1,nl
   do k=1,nz
   do j=1,ny
   do i=1,nx
      if (i >= indxRange(1) .and. i <= indxRange(2) .and. &
          j >= indxRange(3) .and. j <= indxRange(4) .and. &
          k >= indxRange(5) .and. k <= indxRange(6) .and. &
          l >= indxRange(7) .and. l <= indxRange(8) .and. &
          m >= indxRange(9) .and. m <= indxRange(10)) then
         dmask5n(i,j,k,l,m,nfld) = 1.d0
         rmask5n(i,j,k,l,m,nfld) = 1.0
         lmask5n(i,j,k,l,m,nfld) = 1
         imask5n(i,j,k,l,m,nfld) = 1
      else
         dmask5n(i,j,k,l,m,nfld) = 0.d0
         rmask5n(i,j,k,l,m,nfld) = 0.0
         lmask5n(i,j,k,l,m,nfld) = 0
         imask5n(i,j,k,l,m,nfld) = 0
      endif
   end do
   end do
   end do
   end do
   end do
   end do
   ! Use the single-field interface to compute reference sums
   dref = mpas_globalSum(darray5n(:,:,:,:,:,1), MPI_COMM_WORLD)
   rref = mpas_globalSum(rarray5n(:,:,:,:,:,1), MPI_COMM_WORLD)
   iref = mpas_globalSum(iarray5n(:,:,:,:,:,1), MPI_COMM_WORLD)
   lref = mpas_globalSum(larray5n(:,:,:,:,:,1), MPI_COMM_WORLD)

   ! Now compute the various multi-field sums
   dsum1n = mpas_globalSumNfld(darray5n, MPI_COMM_WORLD)
   rsum1n = mpas_globalSumNfld(rarray5n, MPI_COMM_WORLD)
   isum1n = mpas_globalSumNfld(iarray5n, MPI_COMM_WORLD)
   lsum1n = mpas_globalSumNfld(larray5n, MPI_COMM_WORLD)
   dsum2n = mpas_globalSumNfld(darray5n, MPI_COMM_WORLD, indxRange)
   rsum2n = mpas_globalSumNfld(rarray5n, MPI_COMM_WORLD, indxRange)
   isum2n = mpas_globalSumNfld(iarray5n, MPI_COMM_WORLD, indxRange)
   lsum2n = mpas_globalSumNfld(larray5n, MPI_COMM_WORLD, indxRange)
   dsum3n = mpas_globalSumNfld(darray5n, dmask5n, MPI_COMM_WORLD)
   rsum3n = mpas_globalSumNfld(rarray5n, rmask5n, MPI_COMM_WORLD)
   isum3n = mpas_globalSumNfld(iarray5n, imask5n, MPI_COMM_WORLD)
   lsum3n = mpas_globalSumNfld(larray5n, lmask5n, MPI_COMM_WORLD)
   dsum4n = mpas_globalSumNfld(darray5n, dmask5n, MPI_COMM_WORLD, indxRange)
   rsum4n = mpas_globalSumNfld(rarray5n, rmask5n, MPI_COMM_WORLD, indxRange)
   isum4n = mpas_globalSumNfld(iarray5n, imask5n, MPI_COMM_WORLD, indxRange)
   lsum4n = mpas_globalSumNfld(larray5n, lmask5n, MPI_COMM_WORLD, indxRange)

   if (dsum1n(1) == dref .and. dsum1n(2) == dref .and. &
       dsum1n(3) == dref) then
      print *, 'Multi-field and reproducibility 5d double: PASS'
   else
      print *, 'Multi-field and reproducibility 5d double: FAIL', dref, dsum1n
   endif
   if (rsum1n(1) == rref .and. rsum1n(2) == rref .and. &
       rsum1n(3) == rref) then
      print *, 'Multi-field and reproducibility 5d real: PASS'
   else
      print *, 'Multi-field and reproducibility 5d real: FAIL', rref, rsum1n
   endif
   if (isum1n(1) == iref .and. isum1n(2) == iref .and. &
       isum1n(3) == iref) then
      print *, 'Multi-field and reproducibility 5d int: PASS'
   else
      print *, 'Multi-field and reproducibility 5d int: FAIL', iref, isum1n
   endif
   if (lsum1n(1) == lref .and. lsum1n(2) == lref .and. &
       lsum1n(3) == lref) then
      print *, 'Multi-field and reproducibility 5d int(8): PASS'
   else
      print *, 'Multi-field and reproducibility 5d int(8): FAIL', lref, lsum1n
   endif
   isum1 = 0
   isum2 = 0
   isum3 = 0
   isum4 = 0
   do nfld=1,nFields
      if (dsum2n(nfld) /= dsum3n(nfld)) isum1 = isum1+1
      if (dsum2n(nfld) /= dsum4n(nfld)) isum1 = isum1+1
      if (rsum2n(nfld) /= rsum3n(nfld)) isum2 = isum2+1
      if (rsum2n(nfld) /= rsum4n(nfld)) isum2 = isum2+1
      if (isum2n(nfld) /= isum3n(nfld)) isum3 = isum3+1
      if (isum2n(nfld) /= isum4n(nfld)) isum3 = isum3+1
      if (lsum2n(nfld) /= lsum3n(nfld)) isum4 = isum4+1
      if (lsum2n(nfld) /= lsum4n(nfld)) isum4 = isum4+1
   end do
   if (isum1 > 0) then
      print *, 'Multifield range/mask 5d double: FAIL', dsum2n,dsum3n,dsum4n
   else
      print *, 'Multifield range/mask 5d double: PASS'
   endif
   if (isum2 > 0) then
      print *, 'Multifield range/mask 5d real: FAIL', rsum2n,rsum3n,rsum4n
   else
      print *, 'Multifield range/mask 5d real: PASS'
   endif
   if (isum3 > 0) then
      print *, 'Multifield range/mask 5d int: FAIL', isum2n,isum3n,isum4n
   else
      print *, 'Multifield range/mask 5d int: PASS'
   endif
   if (isum4 > 0) then
      print *, 'Multifield range/mask 5d int(8): FAIL', lsum2n,lsum3n,lsum4n
   else
      print *, 'Multifield range/mask 5d int(8): PASS'
   endif
   deallocate(darray5n, dmask5n, rarray5n, rmask5n, &
              iarray5n, imask5n, larray5n, lmask5n)
   deallocate(randoms)

*/
/*
      // test SUM of I4 arrays
      int i, j, k;
      I4 NumCells = 10, NumVertLyrs = 10, C = 0;
      HostArray1DI4 HostArr1DI4("HostArrD1", NumCells);
      HostArray2DI4 HostArr2DI4("HostArrD2", NumCells, NumVertLyrs);
      I4 Sum1DI4 = 0, Sum2DI4 = 0;
      for (i = 0; i < NumCells; i++) {
         HostArr1DI4(i) = i;
         Sum1DI4 += i;
         for (j = 0; j < NumVertLyrs; j++) {
            HostArr2DI4(i, j) = C;
            Sum2DI4 += C;
            ++C;
         }
      }
      globalSum(HostArr1DI4, Comm, &MyResI4);
      expI4 = Sum1DI4 * MySize;
      if (MyResI4 != expI4)
         ABORT_ERROR("Global sum A1DI4: FAIL (exp,act={}, {})", expI4, MyResI4);

      globalSum(HostArr2DI4, Comm, &MyResI4);
      expI4 = Sum2DI4 * MySize;
      if (MyResI4 != expI4)
         ABORT_ERROR("Global sum A2DI4: FAIL (exp,act={}, {})", expI4, MyResI4);

      // test SUM of I8 arrays
      HostArray1DI8 HostArr1DI8("HostArrD1I8", NumCells);
      HostArray2DI8 HostArr2DI8("HostArrD2I8", NumCells, NumVertLyrs);
      I8 Sum1DI8 = 0, Sum2DI8 = 0, C8 = 0;
      for (i = 0; i < NumCells; i++) {
         HostArr1DI8(i) = i;
         Sum1DI8 += i;
         for (j = 0; j < NumVertLyrs; j++) {
            HostArr2DI8(i, j) = C8;
            Sum2DI8 += C8;
            C8++;
         }
      }
      globalSum(HostArr1DI8, Comm, &MyResI8);
      expI8 = Sum1DI8 * MySize;
      if (MyResI8 != expI8)
         ABORT_ERROR("Global sum A1DI8: FAIL (exp,act={}, {})",expI8, MyResI8);

      globalSum(HostArr2DI8, Comm, &MyResI8);
      expI8 = Sum2DI8 * MySize;
      if (MyResI8 != expI8)
         ABORT_ERROR("Global sum A2DI8: FAIL (exp,act={}, {})", expI8, MyResI8);

      // test SUM of R4 arrays
      HostArray1DR4 HostArr1DR4("HostArrD1R4", NumCells);
      HostArray2DR4 HostArr2DR4("HostArrD2R4", NumCells, NumVertLyrs);
      R4 Sum1DR4 = 0.0, Sum2DR4 = 0.0, F = 0.0;
      for (i = 0; i < NumCells; i++) {
         HostArr1DR4(i) = i + 0.00001;
         Sum1DR4 += HostArr1DR4(i);
         for (j = 0; j < NumVertLyrs; j++) {
            HostArr2DR4(i, j) = F;
            Sum2DR4 += HostArr2DR4(i, j);
            F += 1.00001;
         }
      }
      globalSum(HostArr1DR4, Comm, &MyResR4);
      expR4 = Sum1DR4 * MySize;
      if (MyResR4 != expR4)
         ABORT_ERROR("Global sum A1DR4: FAIL (exp,act={}, {})", expR4, MyResR4);

      globalSum(HostArr2DR4, Comm, &MyResR4);
      expR4 = Sum2DR4 * MySize;
      if (MyResR4 != expR4)
         ABORT_ERROR("Global sum A2DR4: FAIL (exp,act={}, {})", expR4, MyResR4);

      // test SUM of R8 arrays
      HostArray1DR8 HostArr1DR8("HostArrD1R8", NumCells);
      HostArray2DR8 HostArr2DR8("HostArrD2R8", NumCells, NumVertLyrs);
      R8 Sum1DR8 = 0.0, Sum2DR8 = 0.0, D = 0.0;
      complex<double> LocalSum1D(0.0, 0.0), LocalSum2D(0.0, 0.0);
      double e, t1, t2, ai;
      for (i = 0; i < NumCells; i++) {
         HostArr1DR8(i) = i + 0.0000000000001;
         // local ddsum
         ai = HostArr1DR8(i);
         t1 = ai + real(LocalSum1D);
         e  = t1 - ai;
         t2 = ((real(LocalSum1D) - e) + (ai - (t1 - e))) + imag(LocalSum1D);
         LocalSum1D = complex<double>(t1 + t2, t2 - ((t1 + t2) - t1));
         for (j = 0; j < NumVertLyrs; j++) {
            HostArr2DR8(i, j) = D;
            D += 1.0000000000001;
            ai = HostArr2DR8(i, j);
            t1 = ai + real(LocalSum2D);
            e  = t1 - ai;
            t2 = ((real(LocalSum2D) - e) + (ai - (t1 - e))) + imag(LocalSum2D);
            LocalSum2D = complex<double>(t1 + t2, t2 - ((t1 + t2) - t1));
         }
      }
      Sum1DR8 = real(LocalSum1D);
      Sum2DR8 = real(LocalSum2D);
      MyResR8 = 0.0;
      globalSum(HostArr1DR8, Comm, &MyResR8);
      // perform serial sum across all MPI tasks
      complex<double> SerialSum(0.0, 0.0);
      for (i = 0; i < MySize; i++) {
         // ddsum across tasks
         t1 = Sum1DR8 + real(SerialSum);
         e  = t1 - Sum1DR8;
         t2 = ((real(SerialSum) - e) + (Sum1DR8 - (t1 - e))) + imag(SerialSum);
         SerialSum = complex<double>(t1 + t2, t2 - ((t1 + t2) - t1));
      }
      expR8 = real(SerialSum);
      if (MyResR8 != expR8)
         ABORT_ERROR("Global sum A1DR8: FAIL (exp,act={}, {})", expR8, MyResR8);

      globalSum(HostArr2DR8, Comm, &MyResR8);
      expR8 = Sum2DR8 * MySize;
      if (MyResR8 != expR8)
         ABORT_ERROR("Global sum A2DR8: FAIL (exp,act={}, {})", expR8, MyResR8);

      //==========================================================================
      // test MIN, MAX of scalars
      MyInt4 = MyTask;
      globalMin(&MyInt4, &MyResI4, Comm);
      if (MyResI4 != 0)
         ABORT_ERROR("Global min I4:    FAIL (exp,act=0, {})", MyResI4);

      MyInt8 = MyTask;
      globalMax(&MyInt8, &MyResI8, Comm);
      if (MyResI8 != MySize - 1)
         ABORT_ERROR("Global max I4:    FAIL (exp,act={}, {})", MySize - 1,
                     MyResI8);

      R8 MyR8Tmp = MyTask + MyR8;
      globalMin(&MyR8Tmp, &MyResR8, Comm);
      if (MyResR8 != MyR8)
         ABORT_ERROR("Global min R8:    FAIL (exp,act={}, {})", MyR8, MyResR8);

      globalMax(&MyR8Tmp, &MyResR8, Comm);
      if (MyResR8 != (MySize - 1 + MyR8))
         ABORT_ERROR("Global max R8:    FAIL (exp,act={}, {})", 
                     MySize - 1 + MyR8, MyResR8);

      //==========================================================================
      // test MIN, MAX of arrays
      HostArray1DI4 HostA1DI4Work("HostA1DI4Work", NumCells * MySize);
      HostArray1DI4 HostA1DI4Min("HostA1DI4Min", NumCells * MySize);
      for (j = 0; j < NumCells; j++) {
         for (i = 0; i < MySize; i++) {
            k = i * NumCells + j;
            if (MyTask != i) {
               HostA1DI4Work(k) = (i + 1) * k; // processor-specific work array
            } else {
               HostA1DI4Work(k) = k; // default
            }
         }
      }
      globalMin(HostA1DI4Work, HostA1DI4Min, Comm);
      int Count = 0;
      for (i = 0; i < NumCells * MySize; i++) {
         if (HostA1DI4Min(i) != i)
            ++Count;
      }
      if (Count > 0)
         ABORT_ERROR("Global min A1DI4: FAIL");

      HostArray1DI4 HostA1DI4Max("HostA1DI4Max", NumCells * MySize);
      globalMax(HostA1DI4Work, HostA1DI4Max, Comm);
      Count = 0;
      for (i = 0; i < MySize; i++) {
         for (j = 0; j < NumCells; j++) {
            k = i * NumCells + j;
            if (HostA1DI4Max(k) != (i + 1) * k)
               ++Count;
         }
      }
      if (Count > 0)
         ABORT_ERROR("Global max A1DI4: FAIL");

      // test SUM of I4 arrays on device
      Array1DI4 DevArr1DI4("DevArr1DI4", NumCells);
      Array2DI4 DevArr2DI4("DevArr2DI4", NumCells, NumVertLyrs);

      parallelFor({NumCells}, KOKKOS_LAMBDA(int i) { DevArr1DI4(i) = i; });
      parallelFor(
          {NumCells, NumVertLyrs},
          KOKKOS_LAMBDA(int i, int j) { DevArr2DI4(i, j) = i * 10 + j; });
      Kokkos::fence();

      globalSum(DevArr1DI4, Comm, &MyResI4);
      expI4 = Sum1DI4 * MySize;
      if (MyResI4 != expI4)
         ABORT_ERROR("Global sum device A1DI4: FAIL (exp,act={}, {})", expI4,
                     MyResI4);

      globalSum(DevArr2DI4, Comm, &MyResI4);
      expI4 = Sum2DI4 * MySize;
      if (MyResI4 != expI4)
         ABORT_ERROR("Global sum device A2DI4: FAIL (exp,act={}, {})", expI4,
                     MyResI4);

      // test SUM of R4 arrays on device
      Array1DR4 DevArr1DR4("DevArrD1R4", NumCells);
      parallelFor(
          {NumCells}, KOKKOS_LAMBDA(int i) { DevArr1DR4(i) = i + 0.00001; });
      Kokkos::fence();

      globalSum(DevArr1DR4, Comm, &MyResR4);
      expR4 = Sum1DR4 * MySize;
      if (MyResR4 != expR4)
         ABORT_ERROR("Global sum device A1DR4: FAIL (exp,act={}, {}", expR4,
                     MyResR4);
*/

      LOG_INFO("------ Global Reductions Unit Tests Sucessful ------");
   }
   Kokkos::finalize();
   MPI_Finalize();

   return 0; // if we made it here, return success

} // end of main
//===-----------------------------------------------------------------------===/
