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

//------------------------------------------------------------------------------
// utility function to compute one iteration of the DD algorithm for
// reproducible double precision sums
void sumDDTest(complex<double> &ddb, double &dda) {
   double t1 = dda + real(ddb);
   double e  = t1 - dda;
   double t2 = ((real(ddb) - e) + (dda - (t1 - e))) + imag(ddb);
   ddb       = complex<double>(t1 + t2, t2 - ((t1 + t2) - t1));
}

//------------------------------------------------------------------------------
// Scalar sum test function
void testScalarSums() {

   // Get some MPI values based on default environment
   MachEnv *DefEnv = MachEnv::getDefault();
   MPI_Comm Comm   = DefEnv->getComm();
   int MyTask      = DefEnv->getMyTask();
   int NTasks      = DefEnv->getNumTasks();

   // Get similar values based on subset environment for reproducibility tests
   MachEnv *SubEnv  = MachEnv::get("Subset");
   MPI_Comm CommSub = SubEnv->getComm();
   int MyTaskSub    = SubEnv->getMyTask();
   int NTasksSub    = SubEnv->getNumTasks();
   bool IsSubMember = SubEnv->isMember();

   // For reproducibility tests, define a full range of values within
   // each data type.  We want the smallest (abs) value > 0
   // Further restrict the max values so that we do not inadvertantly
   // exceed limits.
   I4 MaxEntries = 8;
   I4 MinI4      = 1;
   I4 MaxI4      = std::numeric_limits<I4>::max() / (10 * MaxEntries);
   I8 MinI8      = 1;
   I8 MaxI8      = std::numeric_limits<I8>::max() / (10 * MaxEntries);
   R4 MinR4      = std::numeric_limits<R4>::min();
   R4 MaxR4      = std::numeric_limits<R4>::max() / (10.0 * MaxEntries);
   R8 MinR8      = std::numeric_limits<R8>::min();
   R8 MaxR8      = std::numeric_limits<R8>::max() / (10.0 * MaxEntries);
   R4 EpsR4      = 0.0001;
   R8 EpsR8      = 0.0000000000001;
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

   // Initialize test and reference values
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
   // For reference values, compute a serial sum of all test values on default
   // and subset domains. Use special reproducible sum function for R8.
   double DDValRef;
   double DDValSub;
   complex<double> DDSumRef(0.0, 0.0);
   complex<double> DDSumSub(0.0, 0.0);
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

   // Scalar sum sanity checks
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

   // Test scalar sums with subset communicator
   // Also tests reproducibility using different processor count
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

} // End testScalarSums

//------------------------------------------------------------------------------
// Array sum test function
void testArraySums() {

   // Get some MPI values based on default environment
   MachEnv *DefEnv = MachEnv::getDefault();
   MPI_Comm Comm   = DefEnv->getComm();
   int MyTask      = DefEnv->getMyTask();
   int NTasks      = DefEnv->getNumTasks();

   // Set model size for array tests
   int Nx        = 5;
   int Ny        = 5;
   int Nz        = 3;
   int Nm        = 3;
   int Nn        = 3;
   int NxGlob    = Nx * NTasks;
   I8 MaxEntries = NxGlob * Ny * Nz * Nm * Nn;

   // For reproducibility tests, define a full range of values within
   // each data type.  We want the smallest (abs) value > 0
   // Further restrict the max values so that we do not inadvertantly
   // exceed limits.
   I4 MinI4 = 1;
   I4 MaxI4 = std::numeric_limits<I4>::max() / (10 * MaxEntries);
   I8 MinI8 = 1;
   I8 MaxI8 = std::numeric_limits<I8>::max() / (10 * MaxEntries);
   R4 MinR4 = std::numeric_limits<R4>::min();
   R4 MaxR4 = std::numeric_limits<R4>::max() / (10.0 * MaxEntries);
   R8 MinR8 = std::numeric_limits<R8>::min();
   R8 MaxR8 = std::numeric_limits<R8>::max() / (10.0 * MaxEntries);
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

   // Allocate test arrays
   HostArray1DI4 TestHost1DI4("Host1DI4", Nx);
   HostArray2DI4 TestHost2DI4("Host2DI4", Nx, Ny);
   HostArray3DI4 TestHost3DI4("Host3DI4", Nx, Ny, Nz);
   HostArray4DI4 TestHost4DI4("Host4DI4", Nx, Ny, Nz, Nm);
   HostArray5DI4 TestHost5DI4("Host5DI4", Nx, Ny, Nz, Nm, Nn);
   HostArray1DI8 TestHost1DI8("Host1DI8", Nx);
   HostArray2DI8 TestHost2DI8("Host2DI8", Nx, Ny);
   HostArray3DI8 TestHost3DI8("Host3DI8", Nx, Ny, Nz);
   HostArray4DI8 TestHost4DI8("Host4DI8", Nx, Ny, Nz, Nm);
   HostArray5DI8 TestHost5DI8("Host5DI8", Nx, Ny, Nz, Nm, Nn);
   HostArray1DR4 TestHost1DR4("Host1DR4", Nx);
   HostArray2DR4 TestHost2DR4("Host2DR4", Nx, Ny);
   HostArray3DR4 TestHost3DR4("Host3DR4", Nx, Ny, Nz);
   HostArray4DR4 TestHost4DR4("Host4DR4", Nx, Ny, Nz, Nm);
   HostArray5DR4 TestHost5DR4("Host5DR4", Nx, Ny, Nz, Nm, Nn);
   HostArray1DR8 TestHost1DR8("Host1DR8", Nx);
   HostArray2DR8 TestHost2DR8("Host2DR8", Nx, Ny);
   HostArray3DR8 TestHost3DR8("Host3DR8", Nx, Ny, Nz);
   HostArray4DR8 TestHost4DR8("Host4DR8", Nx, Ny, Nz, Nm);
   HostArray5DR8 TestHost5DR8("Host5DR8", Nx, Ny, Nz, Nm, Nn);
   Array1DI4 Test1DI4("Test1DI4", Nx);
   Array2DI4 Test2DI4("Test2DI4", Nx, Ny);
   Array3DI4 Test3DI4("Test3DI4", Nx, Ny, Nz);
   Array4DI4 Test4DI4("Test4DI4", Nx, Ny, Nz, Nm);
   Array5DI4 Test5DI4("Test5DI4", Nx, Ny, Nz, Nm, Nn);
   Array1DI8 Test1DI8("Test1DI8", Nx);
   Array2DI8 Test2DI8("Test2DI8", Nx, Ny);
   Array3DI8 Test3DI8("Test3DI8", Nx, Ny, Nz);
   Array4DI8 Test4DI8("Test4DI8", Nx, Ny, Nz, Nm);
   Array5DI8 Test5DI8("Test5DI8", Nx, Ny, Nz, Nm, Nn);
   Array1DR4 Test1DR4("Test1DR4", Nx);
   Array2DR4 Test2DR4("Test2DR4", Nx, Ny);
   Array3DR4 Test3DR4("Test3DR4", Nx, Ny, Nz);
   Array4DR4 Test4DR4("Test4DR4", Nx, Ny, Nz, Nm);
   Array5DR4 Test5DR4("Test5DR4", Nx, Ny, Nz, Nm, Nn);
   Array1DR8 Test1DR8("Test1DR8", Nx);
   Array2DR8 Test2DR8("Test2DR8", Nx, Ny);
   Array3DR8 Test3DR8("Test3DR8", Nx, Ny, Nz);
   Array4DR8 Test4DR8("Test4DR8", Nx, Ny, Nz, Nm);
   Array5DR8 Test5DR8("Test5DR8", Nx, Ny, Nz, Nm, Nn);

   // Compute reference values using a serial reproducible sum
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
         DDVal1D = (IGlob + EpsR8) * FacR8[Task];
         sumDDTest(DDSumRef1D, DDVal1D); // local repro sum
         for (int J = 0; J < Ny; ++J) {
            int Jindx = IGlob + J;
            Ref2DI4 += Jindx * FacI4[Task];
            Ref2DI8 += Jindx * FacI8[Task];
            Tmp2DR4 += (Jindx + EpsR4) * FacR4[Task];
            DDVal2D = (Jindx + EpsR8) * FacR8[Task];
            sumDDTest(DDSumRef2D, DDVal2D); // local repro sum
            for (int K = 0; K < Nz; ++K) {
               int Kindx = IGlob + J + K;
               Ref3DI4 += Kindx * FacI4[Task];
               Ref3DI8 += Kindx * FacI8[Task];
               Tmp3DR4 += (Kindx + EpsR4) * FacR4[Task];
               DDVal3D = (Kindx + EpsR8) * FacR8[Task];
               sumDDTest(DDSumRef3D, DDVal3D); // local repro sum
               for (int M = 0; M < Nm; ++M) {
                  int Mindx = IGlob + J + K + M;
                  Ref4DI4 += Mindx * FacI4[Task];
                  Ref4DI8 += Mindx * FacI8[Task];
                  Tmp4DR4 += (Mindx + EpsR4) * FacR4[Task];
                  DDVal4D = (Mindx + EpsR8) * FacR8[Task];
                  sumDDTest(DDSumRef4D, DDVal4D); // local repro sum
                  for (int N = 0; N < Nn; ++N) {
                     int Nindx = IGlob + J + K + M + N;
                     Ref5DI4 += Nindx * FacI4[Task];
                     Ref5DI8 += Nindx * FacI8[Task];
                     Tmp5DR4 += (Nindx + EpsR4) * FacR4[Task];
                     DDVal5D = (Nindx + EpsR8) * FacR8[Task];
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
      int IGlobal     = MyTask * Nx + I;
      TestHost1DI4(I) = IGlobal * FacI4[MyTask];
      TestHost1DI8(I) = IGlobal * FacI8[MyTask];
      TestHost1DR4(I) = (IGlobal + EpsR4) * FacR4[MyTask];
      TestHost1DR8(I) = (IGlobal + EpsR8) * FacR8[MyTask];
      for (int J = 0; J < Ny; ++J) {
         int JIndx          = IGlobal + J;
         TestHost2DI4(I, J) = JIndx * FacI4[MyTask];
         TestHost2DI8(I, J) = JIndx * FacI8[MyTask];
         TestHost2DR4(I, J) = (JIndx + EpsR4) * FacR4[MyTask];
         TestHost2DR8(I, J) = (JIndx + EpsR8) * FacR8[MyTask];
         for (int K = 0; K < Nz; ++K) {
            int KIndx             = IGlobal + J + K;
            TestHost3DI4(I, J, K) = KIndx * FacI4[MyTask];
            TestHost3DI8(I, J, K) = KIndx * FacI8[MyTask];
            TestHost3DR4(I, J, K) = (KIndx + EpsR4) * FacR4[MyTask];
            TestHost3DR8(I, J, K) = (KIndx + EpsR8) * FacR8[MyTask];
            for (int M = 0; M < Nm; ++M) {
               int MIndx                = IGlobal + J + K + M;
               TestHost4DI4(I, J, K, M) = MIndx * FacI4[MyTask];
               TestHost4DI8(I, J, K, M) = MIndx * FacI8[MyTask];
               TestHost4DR4(I, J, K, M) = (MIndx + EpsR4) * FacR4[MyTask];
               TestHost4DR8(I, J, K, M) = (MIndx + EpsR8) * FacR8[MyTask];
               for (int N = 0; N < Nn; ++N) {
                  int NIndx                   = IGlobal + J + K + M + N;
                  TestHost5DI4(I, J, K, M, N) = NIndx * FacI4[MyTask];
                  TestHost5DI8(I, J, K, M, N) = NIndx * FacI8[MyTask];
                  TestHost5DR4(I, J, K, M, N) = (NIndx + EpsR4) * FacR4[MyTask];
                  TestHost5DR8(I, J, K, M, N) = (NIndx + EpsR8) * FacR8[MyTask];
               }
            }
         }
      }
   }

   // Copy values to device test arrays
   deepCopy(Test1DI4, TestHost1DI4);
   deepCopy(Test2DI4, TestHost2DI4);
   deepCopy(Test3DI4, TestHost3DI4);
   deepCopy(Test4DI4, TestHost4DI4);
   deepCopy(Test5DI4, TestHost5DI4);
   deepCopy(Test1DI8, TestHost1DI8);
   deepCopy(Test2DI8, TestHost2DI8);
   deepCopy(Test3DI8, TestHost3DI8);
   deepCopy(Test4DI8, TestHost4DI8);
   deepCopy(Test5DI8, TestHost5DI8);
   deepCopy(Test1DR4, TestHost1DR4);
   deepCopy(Test2DR4, TestHost2DR4);
   deepCopy(Test3DR4, TestHost3DR4);
   deepCopy(Test4DR4, TestHost4DR4);
   deepCopy(Test5DR4, TestHost5DR4);
   deepCopy(Test1DR8, TestHost1DR8);
   deepCopy(Test2DR8, TestHost2DR8);
   deepCopy(Test3DR8, TestHost3DR8);
   deepCopy(Test4DR8, TestHost4DR8);
   deepCopy(Test5DR8, TestHost5DR8);

   //-----
   // Test full array interface and sums
   I4 SumI4 = 0;
   I8 SumI8 = 0;
   R4 SumR4 = 0.0;
   R8 SumR8 = 0.0;

   //-----
   // Compute 1D full host array sums and perform error checks
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

   // Compute 1D full device array sums and perform error checks
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
   // if (SumR8 != Ref1DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 array)"
   //                "Expected = {} Actual = {}",
   //                Ref1DR8, SumR8);

   // Compute 2D full host array sums and perform error checks
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

   // Compute 3D full device array sums and perform error checks
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
   // if (SumR8 != Ref2DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 array)"
   //                "Expected = {} Actual = {}",
   //                Ref2DR8, SumR8);

   // Compute 3D full host array sums and perform error checks
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

   // Compute 3D full device array sums and perform error checks
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
   // if (SumR8 != Ref3DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 array)"
   //                "Expected = {} Actual = {}",
   //                Ref3DR8, SumR8);

   // Compute 4D full host array sums and perform error checks
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

   // Compute 4D full device array sums and perform error checks
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
   // if (SumR8 != Ref4DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 array)"
   //                "Expected = {} Actual = {}",
   //                Ref4DR8, SumR8);

   // Compute 5D full host array sums and perform error checks
   LOG_ERROR("Before Host5DI4 sum");
   SumI4 = globalSum(TestHost5DI4, Comm);
   LOG_ERROR("Barrier Host5DI4");
   SumI8 = globalSum(TestHost5DI8, Comm);
   LOG_ERROR("Barrier Host5DI8");
   SumR4 = globalSum(TestHost5DR4, Comm);
   LOG_ERROR("Barrier Host5DR4");
   SumR8 = globalSum(TestHost5DR8, Comm);
   LOG_ERROR("Barrier Host5DR8");
   LOG_INFO("Array5D host sum I4 {} {}", SumI4, Ref5DI4);
   LOG_ERROR("Barrier");
   LOG_INFO("Array5D host sum I8 {} {}", SumI8, Ref5DI8);
   LOG_ERROR("Barrier");
   LOG_INFO("Array5D host sum R4 {} {}", SumR4, Ref5DR4);
   LOG_ERROR("Barrier");
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

   // Compute 5D full device array sums and perform error checks
   SumI4 = globalSum(Test5DI4, Comm);
   LOG_ERROR("Barrier Dev5DI4");
   SumI8 = globalSum(Test5DI8, Comm);
   LOG_ERROR("Barrier Dev5DI8");
   SumR4 = globalSum(Test5DR4, Comm);
   LOG_ERROR("Barrier Dev5DR4");
   SumR8 = globalSum(Test5DR8, Comm);
   LOG_ERROR("Barrier Dev5DR8");
   LOG_INFO("Array5D dev sum I4 {} {}", SumI4, Ref5DI4);
   LOG_ERROR("Barrier");
   LOG_INFO("Array5D dev sum I8 {} {}", SumI8, Ref5DI8);
   LOG_ERROR("Barrier");
   LOG_INFO("Array5D dev sum R4 {} {}", SumR4, Ref5DR4);
   LOG_ERROR("Barrier");
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
   // if (SumR8 != Ref5DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 array)"
   //                "Expected = {} Actual = {}",
   //                Ref5DR8, SumR8);

   //-----
   // Test arrays with index range and restrictions and test sums with product
   // by creating a mask that is only 1 where index range is valid and compare.
   //-----

   // Initialize mask arrays
   HostArray1DI4 MaskHost1DI4("MaskH1DI4", Nx);
   HostArray1DI8 MaskHost1DI8("MaskH1DI8", Nx);
   HostArray1DR4 MaskHost1DR4("MaskH1DR4", Nx);
   HostArray1DR8 MaskHost1DR8("MaskH1DR8", Nx);
   HostArray2DI4 MaskHost2DI4("MaskH2DI4", Nx, Ny);
   HostArray2DI8 MaskHost2DI8("MaskH2DI8", Nx, Ny);
   HostArray2DR4 MaskHost2DR4("MaskH2DR4", Nx, Ny);
   HostArray2DR8 MaskHost2DR8("MaskH2DR8", Nx, Ny);
   HostArray3DI4 MaskHost3DI4("MaskH3DI4", Nx, Ny, Nz);
   HostArray3DI8 MaskHost3DI8("MaskH3DI8", Nx, Ny, Nz);
   HostArray3DR4 MaskHost3DR4("MaskH3DR4", Nx, Ny, Nz);
   HostArray3DR8 MaskHost3DR8("MaskH3DR8", Nx, Ny, Nz);
   HostArray4DI4 MaskHost4DI4("MaskH4DI4", Nx, Ny, Nz, Nm);
   HostArray4DI8 MaskHost4DI8("MaskH4DI8", Nx, Ny, Nz, Nm);
   HostArray4DR4 MaskHost4DR4("MaskH4DR4", Nx, Ny, Nz, Nm);
   HostArray4DR8 MaskHost4DR8("MaskH4DR8", Nx, Ny, Nz, Nm);
   HostArray5DI4 MaskHost5DI4("MaskH5DI4", Nx, Ny, Nz, Nm, Nn);
   HostArray5DI8 MaskHost5DI8("MaskH5DI8", Nx, Ny, Nz, Nm, Nn);
   HostArray5DR4 MaskHost5DR4("MaskH5DR4", Nx, Ny, Nz, Nm, Nn);
   HostArray5DR8 MaskHost5DR8("MaskH5DR8", Nx, Ny, Nz, Nm, Nn);
   Array1DI4 Mask1DI4("Mask1DI4", Nx);
   Array1DI8 Mask1DI8("Mask1DI8", Nx);
   Array1DR4 Mask1DR4("Mask1DR4", Nx);
   Array1DR8 Mask1DR8("Mask1DR8", Nx);
   Array2DI4 Mask2DI4("Mask2DI4", Nx, Ny);
   Array2DI8 Mask2DI8("Mask2DI8", Nx, Ny);
   Array2DR4 Mask2DR4("Mask2DR4", Nx, Ny);
   Array2DR8 Mask2DR8("Mask2DR8", Nx, Ny);
   Array3DI4 Mask3DI4("Mask3DI4", Nx, Ny, Nz);
   Array3DI8 Mask3DI8("Mask3DI8", Nx, Ny, Nz);
   Array3DR4 Mask3DR4("Mask3DR4", Nx, Ny, Nz);
   Array3DR8 Mask3DR8("Mask3DR8", Nx, Ny, Nz);
   Array4DI4 Mask4DI4("Mask4DI4", Nx, Ny, Nz, Nm);
   Array4DI8 Mask4DI8("Mask4DI8", Nx, Ny, Nz, Nm);
   Array4DR4 Mask4DR4("Mask4DR4", Nx, Ny, Nz, Nm);
   Array4DR8 Mask4DR8("Mask4DR8", Nx, Ny, Nz, Nm);
   Array5DI4 Mask5DI4("Mask5DI4", Nx, Ny, Nz, Nm, Nn);
   Array5DI8 Mask5DI8("Mask5DI8", Nx, Ny, Nz, Nm, Nn);
   Array5DR4 Mask5DR4("Mask5DR4", Nx, Ny, Nz, Nm, Nn);
   Array5DR8 Mask5DR8("Mask5DR8", Nx, Ny, Nz, Nm, Nn);

   // Set restricted index range
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
   // Reset various sums and compute new reference values
   Ref1DI4    = 0;
   Ref1DI8    = 0;
   Ref1DR4    = 0.0;
   Ref1DR8    = 0.0;
   Tmp1DR4    = 0.0;
   DDVal1D    = 0.0;
   Ref2DI4    = 0;
   Ref2DI8    = 0;
   Ref2DR4    = 0.0;
   Ref2DR8    = 0.0;
   Tmp2DR4    = 0.0;
   DDVal2D    = 0.0;
   Ref3DI4    = 0;
   Ref3DI8    = 0;
   Ref3DR4    = 0.0;
   Ref3DR8    = 0.0;
   Tmp3DR4    = 0.0;
   DDVal3D    = 0.0;
   Ref4DI4    = 0;
   Ref4DI8    = 0;
   Ref4DR4    = 0.0;
   Ref4DR8    = 0.0;
   Tmp4DR4    = 0.0;
   DDVal4D    = 0.0;
   Ref5DI4    = 0;
   Ref5DI8    = 0;
   Ref5DR4    = 0.0;
   Ref5DR8    = 0.0;
   Tmp5DR4    = 0.0;
   DDVal5D    = 0.0;
   DDSumRef1D = complex<double>(0.0, 0.0);
   DDSumRef2D = complex<double>(0.0, 0.0);
   DDSumRef3D = complex<double>(0.0, 0.0);
   DDSumRef4D = complex<double>(0.0, 0.0);
   DDSumRef5D = complex<double>(0.0, 0.0);
   // Compute new reference sums for restricted range
   for (int Task = 0; Task < NTasks; ++Task) {
      for (int I = IMin; I <= IMax; ++I) {
         int IGlob = Task * Nx + I;
         Ref1DI4 += IGlob * FacI4[Task];
         Ref1DI8 += IGlob * FacI8[Task];
         Tmp1DR4 += (IGlob + EpsR4) * FacR4[Task];
         DDVal1D = (IGlob + EpsR8) * FacR8[Task];
         sumDDTest(DDSumRef1D, DDVal1D); // local repro sum
         for (int J = JMin; J <= JMax; ++J) {
            int Jindx = IGlob + J;
            Ref2DI4 += Jindx * FacI4[Task];
            Ref2DI8 += Jindx * FacI8[Task];
            Tmp2DR4 += (Jindx + EpsR4) * FacR4[Task];
            DDVal2D = (Jindx + EpsR8) * FacR8[Task];
            sumDDTest(DDSumRef2D, DDVal2D); // local repro sum
            for (int K = KMin; K <= KMax; ++K) {
               int Kindx = IGlob + J + K;
               Ref3DI4 += Kindx * FacI4[Task];
               Ref3DI8 += Kindx * FacI8[Task];
               Tmp3DR4 += (Kindx + EpsR4) * FacR4[Task];
               DDVal3D = (Kindx + EpsR8) * FacR8[Task];
               sumDDTest(DDSumRef3D, DDVal3D); // local repro sum
               for (int M = MMin; M <= MMax; ++M) {
                  int Mindx = IGlob + J + K + M;
                  Ref4DI4 += Mindx * FacI4[Task];
                  Ref4DI8 += Mindx * FacI8[Task];
                  Tmp4DR4 += (Mindx + EpsR4) * FacR4[Task];
                  DDVal4D = (Mindx + EpsR8) * FacR8[Task];
                  sumDDTest(DDSumRef4D, DDVal4D); // local repro sum
                  for (int N = NMin; N <= NMax; ++N) {
                     int Nindx = IGlob + J + K + M + N;
                     Ref5DI4 += Nindx * FacI4[Task];
                     Ref5DI8 += Nindx * FacI8[Task];
                     Tmp5DR4 += (Nindx + EpsR4) * FacR4[Task];
                     DDVal5D = (Nindx + EpsR8) * FacR8[Task];
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

   // Fill mask arrays on host
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
   deepCopy(Mask1DI4, MaskHost1DI4);
   deepCopy(Mask1DI8, MaskHost1DI8);
   deepCopy(Mask1DR4, MaskHost1DR4);
   deepCopy(Mask1DR8, MaskHost1DR8);
   deepCopy(Mask2DI4, MaskHost2DI4);
   deepCopy(Mask2DI8, MaskHost2DI8);
   deepCopy(Mask2DR4, MaskHost2DR4);
   deepCopy(Mask2DR8, MaskHost2DR8);
   deepCopy(Mask3DI4, MaskHost3DI4);
   deepCopy(Mask3DI8, MaskHost3DI8);
   deepCopy(Mask3DR4, MaskHost3DR4);
   deepCopy(Mask3DR8, MaskHost3DR8);
   deepCopy(Mask4DI4, MaskHost4DI4);
   deepCopy(Mask4DI8, MaskHost4DI8);
   deepCopy(Mask4DR4, MaskHost4DR4);
   deepCopy(Mask4DR8, MaskHost4DR8);
   deepCopy(Mask5DI4, MaskHost5DI4);
   deepCopy(Mask5DI8, MaskHost5DI8);
   deepCopy(Mask5DR4, MaskHost5DR4);
   deepCopy(Mask5DR8, MaskHost5DR8);

   //-----
   // Compute sums with range limits
   // Compute 1D host array sums with address range
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

   // Compute 1D device array sums with address range
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
   // if (SumR8 != Ref1DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 array address
   //    range)"
   //                "Expected = {} Actual = {}",
   //                Ref1DR8, SumR8);

   // Compute 2D host array sums with address range
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

   // Compute 2D device array sums with address range
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
   // if (SumR8 != Ref2DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 array address
   //    range)"
   //                "Expected = {} Actual = {}",
   //                Ref2DR8, SumR8);

   // Compute 3D host array sums with address range
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

   // Compute 3D device array sums with address range
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
   // if (SumR8 != Ref3DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 array address
   //    range)"
   //                "Expected = {} Actual = {}",
   //                Ref3DR8, SumR8);

   // Compute 4D host array sums with address range
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

   // Compute 4D device array sums with address range
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
   // if (SumR8 != Ref4DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 array address
   //    range)"
   //                "Expected = {} Actual = {}",
   //                Ref4DR8, SumR8);

   // Compute 5D host array sums with address range
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

   // Compute 5D device array sums with address range
   // LOG_ERROR("Before 5d range tests I4");
   SumI4 = globalSum(Test5DI4, Comm, &AddRange);
   // LOG_ERROR("Before 5d range tests I8");
   SumI8 = globalSum(Test5DI8, Comm, &AddRange);
   // LOG_ERROR("Before 5d range tests R4");
   SumR4 = globalSum(Test5DR4, Comm, &AddRange);
   // LOG_ERROR("Before 5d range tests R8");
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
   // if (SumR8 != Ref5DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 array address
   //    range)"
   //                "Expected = {} Actual = {}",
   //                Ref5DR8, SumR8);

   //-----
   // Sums with product - full arrays

   // Test sum with product for full 1D host arrays
   // LOG_ERROR("Before sum with products I4");
   SumI4 = globalSum(TestHost1DI4, MaskHost1DI4, Comm);
   // LOG_ERROR("Before sum with products I8");
   SumI8 = globalSum(TestHost1DI8, MaskHost1DI8, Comm);
   // LOG_ERROR("Before sum with products R4");
   SumR4 = globalSum(TestHost1DR4, MaskHost1DR4, Comm);
   // LOG_ERROR("Before sum with products R8");
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

   // Test sum with product for full 1D device arrays
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
   // if (SumR8 != Ref1DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (1DR8 sum product)"
   //                "Expected = {} Actual = {}",
   //                Ref1DR8, SumR8);

   // Test sum with product for full 2D host arrays
   SumI4 = globalSum(TestHost2DI4, MaskHost2DI4, Comm);
   SumI8 = globalSum(TestHost2DI8, MaskHost2DI8, Comm);
   SumR4 = globalSum(TestHost2DR4, MaskHost2DR4, Comm);
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

   // Test sum with product for full 2D device arrays
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
   // if (SumR8 != Ref2DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (2DR8 sum product)"
   //                "Expected = {} Actual = {}",
   //                Ref2DR8, SumR8);

   // Test sum with product for full 3D host arrays
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

   // Test sum with product for full 3D device arrays
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
   // if (SumR8 != Ref3DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (3DR8 sum product)"
   //                "Expected = {} Actual = {}",
   //                Ref3DR8, SumR8);

   // Test sum with product for full 4D host arrays
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

   // Test sum with product for full 4D device arrays
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
   // if (SumR8 != Ref4DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (4DR8 sum product)"
   //                "Expected = {} Actual = {}",
   //                Ref4DR8, SumR8);

   // Test sum with product for full 5D host arrays
   SumI4 = globalSum(TestHost5DI4, MaskHost5DI4, Comm);
   SumI8 = globalSum(TestHost5DI8, MaskHost5DI8, Comm);
   SumR4 = globalSum(TestHost5DR4, MaskHost5DR4, Comm);
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

   // Test sum with product for full 5D device arrays
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
   // if (SumR8 != Ref5DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum (5DR8 sum product)"
   //                "Expected = {} Actual = {}",
   //                Ref5DR8, SumR8);

   //------
   // Sum with product - subset range
   // Test sum with product and index range for 1D host arrays
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

   // Test sum with product and index range for 1D device arrays
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
   // if (SumR8 != Ref1DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum "
   //                "(1DR8 sum with product and address range)"
   //                "Expected = {} Actual = {}",
   //                Ref1DR8, SumR8);

   // Test sum with product and index range for 2D host arrays
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

   // Test sum with product and index range for 2D device arrays
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
   // if (SumR8 != Ref2DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum "
   //                "(2DR8 sum with product and address range)"
   //                "Expected = {} Actual = {}",
   //                Ref2DR8, SumR8);

   // Test sum with product and index range for 3D host arrays
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

   // Test sum with product and index range for 3D device arrays
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
   // if (SumR8 != Ref3DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum "
   //                "(3DR8 sum with product and address range)"
   //                "Expected = {} Actual = {}",
   //                Ref3DR8, SumR8);

   // Test sum with product and index range for 4D host arrays
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

   // Test sum with product and index range for 4D device arrays
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
   // if (SumR8 != Ref4DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum "
   //                "(4DR8 sum with product and address range)"
   //                "Expected = {} Actual = {}",
   //                Ref4DR8, SumR8);

   // Test sum with product and index range for 5D host arrays
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

   // Test sum with product and index range for 5D device arrays
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
   // if (SumR8 != Ref5DR8)
   //    ABORT_ERROR("ReductionsTest: FAIL globalSum "
   //                "(5DR8 sum with product and address range)"
   //                "Expected = {} Actual = {}",
   //                Ref5DR8, SumR8);

   LOG_ERROR("At end of testArraySums");

} // End testArraySums

//------------------------------------------------------------------------------
// Main test driver
int main(int argc, char *argv[]) {

   // Initialize various environments and utilities
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv = MachEnv::getDefault();
   Pacer::initialize(MPI_COMM_WORLD);
   Pacer::setPrefix("Omega:");
   OMEGA::initLogging(DefEnv);
   LOG_INFO("------ Global Reductions Unit Tests ------");

   // For reproducibility across tasks, create a smaller sub-environment
   int NTasks = DefEnv->getNumTasks();
   if (NTasks < 8)
      ABORT_ERROR(
          "ReductionsTest: FAIL must run unit test with at least 8 tasks");
   MachEnv::create("Subset", DefEnv, 4); // contiguous subset environment

   // Call individual test routines
   testScalarSums();
   testArraySums();
   // testMinMax();

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

   LOG_INFO("------ Global Reductions Unit Tests Sucessful ------");

   // clean up
   LOG_ERROR("cleanup");
   MachEnv::removeAll();
   LOG_ERROR("Kokkos Finalize");
   Kokkos::finalize();
   LOG_ERROR("MPI Finalize");
   MPI_Finalize();

   return 0; // if we made it here, return success

} // end of main
//===-----------------------------------------------------------------------===/
