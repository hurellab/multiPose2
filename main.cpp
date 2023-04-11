#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <algorithm>
#include <igl/readTGF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeDMAT.h>
#include <igl/writeMESH.h>
#include <igl/readDMAT.h>
#include <igl/dqs.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/boundary_conditions.h>
#include <igl/in_element.h>
#include <igl/bbw.h>
#include <Eigen/Dense>

typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>
    RotationList;
typedef std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
    MatrixList;
typedef std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
    Matrix3dList;

void myDqs(
  const Eigen::MatrixXd & V,
  const std::vector<std::map<int, double>> & W,
  const RotationList & vQ,
  const std::vector<Eigen::Vector3d> & vT,
  Eigen::MatrixXd & U)
{
  using namespace std;
  U.resizeLike(V);

  // Convert quats + trans into dual parts
  vector<Eigen::Quaterniond> vD(vQ.size());
  for(size_t c = 0;c<vQ.size();c++)
  {
    const Eigen::Quaterniond  & q = vQ[c];
    vD[c].w() = -0.5*( vT[c](0)*q.x() + vT[c](1)*q.y() + vT[c](2)*q.z());
    vD[c].x() =  0.5*( vT[c](0)*q.w() + vT[c](1)*q.z() - vT[c](2)*q.y());
    vD[c].y() =  0.5*(-vT[c](0)*q.z() + vT[c](1)*q.w() + vT[c](2)*q.x());
    vD[c].z() =  0.5*( vT[c](0)*q.y() - vT[c](1)*q.x() + vT[c](2)*q.w());
  }

  // Loop over vertices
  const int nv = V.rows();
#pragma omp parallel for if (nv>10000)
  for(int i = 0;i<nv;i++)
  {
    Eigen::Quaterniond b0(0,0,0,0);
    Eigen::Quaterniond be(0,0,0,0);
    Eigen::Quaterniond vQ0;
    bool first(true);
    // Loop over handles
    for(auto iter:W[i])
    {
        if(first){
            b0.coeffs() = iter.second * vQ[iter.first].coeffs();
            be.coeffs() = iter.second * vD[iter.first].coeffs();
            vQ0 = vQ[iter.first];
            first = false;
            continue;
        }
        if( vQ0.dot( vQ[iter.first] ) < 0.f ){
            b0.coeffs() -= iter.second * vQ[iter.first].coeffs();
            be.coeffs() -= iter.second * vD[iter.first].coeffs();
        }else{
            b0.coeffs() += iter.second * vQ[iter.first].coeffs();
            be.coeffs() += iter.second * vD[iter.first].coeffs();
        }
    }
    Eigen::Quaterniond ce = be;
    ce.coeffs() /= b0.norm();
    Eigen::Quaterniond c0 = b0;
    c0.coeffs() /= b0.norm();
    // See algorithm 1 in "Geometric skinning with approximate dual quaternion
    // blending" by Kavan et al
    Eigen::Vector3d v = V.row(i);
    Eigen::Vector3d d0 = c0.vec();
    Eigen::Vector3d de = ce.vec();
    Eigen::Quaterniond::Scalar a0 = c0.w();
    Eigen::Quaterniond::Scalar ae = ce.w();
    U.row(i) =  v + 2*d0.cross(d0.cross(v) + a0*v) + 2*(a0*de - ae*d0 + d0.cross(de));
  }
}

Eigen::SparseMatrix<double> GenBary(const Eigen::MatrixXd &V, const Eigen::MatrixXd &VT, const Eigen::MatrixXi &TT)
{
   Eigen::SparseMatrix<double> bary;
   std::cout << "AABB->" << std::flush;
   igl::AABB<Eigen::MatrixXd, 3> tree;
   tree.init(VT, TT);
   Eigen::VectorXi I;
   std::cout << "in_element->" << std::flush;
   igl::in_element(VT, TT, V, tree, I);
   std::cout << "calculate vol." << std::endl;
   Eigen::VectorXd invol;
   igl::volume(VT, TT, invol);
   invol = -(invol * 6.).cwiseInverse();
   typedef Eigen::Triplet<double> T;
   std::vector<T> coeff;
   bary.resize(V.rows(), VT.rows());
   if((I.array()<0).cast<int>().sum()>0)
    std::cout<<"out tet --> "<< (I.array()<0).cast<int>().sum() << std::endl;
   for (int i = 0; i < I.rows(); i++)
   {
    // if(I(i)<0) std::cout<<"!!!"<<std::endl;
     std::vector<Eigen::Vector3d> tet;
     for (int n = 0; n < 4; n++)
       tet.push_back(Eigen::Vector3d(VT.row(TT(I(i), n))));
     Eigen::Vector3d v = V.row(i);
     coeff.push_back(T(i, TT(I(i), 0), (v - tet[3]).cross(tet[1] - tet[3]).dot(tet[2] - tet[3]) * invol(I(i))));
     coeff.push_back(T(i, TT(I(i), 1), (tet[0] - tet[3]).cross(v - tet[3]).dot(tet[2] - tet[3]) * invol(I(i))));
     coeff.push_back(T(i, TT(I(i), 2), (tet[0] - tet[3]).cross(tet[1] - tet[3]).dot(v - tet[3]) * invol(I(i))));
     coeff.push_back(T(i, TT(I(i), 3), (tet[0] - v).cross(tet[1] - v).dot(tet[2] - v) * invol(I(i))));
     std::cout << "\rGenerating barycentric_coordinate..." << i << "/" << I.rows() - 1 << std::flush;
   }
   bary.setFromTriplets(coeff.begin(), coeff.end());
   std::cout << std::endl;
   return bary;
}

void PrintUsage(){
  std::cout<<"./poseEst [phantom name]"<<std::endl;
}
int main(int argc, char *argv[])
{
  if(argc < 2)
  {
    PrintUsage();
    return 1;
  }
  /////////////////////////phantom import//////////////////////////////
  // TGF
  Eigen::MatrixXd C;
  Eigen::MatrixXi BE;
  std::string phantom(argv[1]);
  if(!igl::readTGF(phantom + ".tgf", C, BE))
  {
    std::cout<<"There is no "+phantom+".tgf"<<std::endl;
    return 1;
  }
  double rootL = (C.row(16)-C.row(9)).norm() / (C.row(6)-C.row(9)).norm();
  double shoulM = 1.-(C.row(19)-C.row(0)).dot((C.row(3)-C.row(0)).normalized()) / (C.row(3)-C.row(0)).norm();

  // EXTRACT SHELL
  Eigen::MatrixXd Vo;
  Eigen::MatrixXi Fo;
  std::vector<Eigen::RowVector3d> Vtmp;
  std::vector<Eigen::RowVector3i> Ftmp;
  std::set<std::string> external = {"12200_Skin_surface", "6700_Cornea_left", "6900_Cornea_right"};
  std::ifstream ifsObj(phantom+".obj");
  if(!ifsObj.is_open())
  {
    std::cout<<"There is no "+phantom+".obj"<<std::endl;
    return 1;
  }
  bool extract(false);
  int numV(0), numVo(0);
  while(!ifsObj.eof())
  {
    std::string aLine, first;
    std::getline(ifsObj,aLine);
    std::stringstream ss(aLine);
    ss>>first;
    if(first=="v")
    {
      double x, y, z;
      ss>>x>>y>>z;
      Vtmp.push_back(Eigen::RowVector3d(x,y,z));
    }
    else if(extract && first=="f")
    {
      int a, b, c;
      ss>>a>>b>>c;
      Ftmp.push_back(Eigen::RowVector3i(a,b,c));
    }
    else if(first=="g")
    {
      if(Ftmp.size())
      {
        Eigen::MatrixXi Fo1(Ftmp.size(), 3);
        for(size_t i=0;i<Ftmp.size();i++) Fo1.row(i) = Ftmp[i];
        Fo.conservativeResize(Fo.rows() + Fo1.rows(), 3);
        Fo.bottomRows(Fo1.rows()) = Fo1.array() + numVo;
        Ftmp.clear();
      }
      std::string shellName;
      ss>>shellName;
      std::cout<<"reading "+shellName<<std::endl;
      if(external.find(shellName) != external.end())
      {
        extract = true;
        numVo = Vo.rows() - numV -1;
        Eigen::MatrixXd Vo1(Vtmp.size(), 3);
        for(size_t i=0;i<Vtmp.size();i++) Vo1.row(i) = Vtmp[i];
        Vo.conservativeResize(Vo.rows() + Vo1.rows(), 3);
        Vo.bottomRows(Vo1.rows()) = Vo1;
      }
      else extract = false;
      numV += Vtmp.size();
      Vtmp.clear();
    }
  }
  Ftmp.clear();
  ifsObj.close();
  numVo = Vo.rows();
  Eigen::MatrixXd VT, W, Wj;
  Eigen::MatrixXi TT;
  // Eigen::VectorXi G(BE.rows());
  // G<<0, 1, 2, 15, 2, 2, 3, 4, 5, 2, 2, 6, 7, 8, 0, 9, 10, 11, 0, 12, 13, 14;
  if(!igl::readDMAT(phantom+"_ELE.dmat", TT) ||
     !igl::readDMAT(phantom+"_NODE.dmat", VT) || 
     !igl::readDMAT(phantom+"_j.dmat", Wj) || 
     !igl::readDMAT(phantom+"_b.dmat", W))
  {
    // TETRAHEDRALIZATION
    Vo.conservativeResize(Vo.rows() + C.rows(), 3);
    Vo.bottomRows(C.rows()) = C;
    for (int i = 0; i < BE.rows(); i++)
    {
      int num = (C.row(BE(i, 0)) - C.row(BE(i, 1))).norm();
      Eigen::RowVector3d itvl = (C.row(BE(i, 1)) - C.row(BE(i, 0))) / (double)num;
      Eigen::MatrixXd boneP(num - 1, 3);
      for (int n = 1; n < num; n++)
        boneP.row(n - 1) = C.row(BE(i, 0)) + n * itvl;
      Vo.conservativeResize(Vo.rows() + num - 1, 3);
      Vo.bottomRows(num - 1) = boneP;
    }
    Eigen::MatrixXi FT;
    igl::copyleft::tetgen::tetrahedralize(Vo, Fo, "qp/0.0001YT0.000000001", VT, TT, FT);
    FT.resize(0, 0);
    igl::writeDMAT(phantom+"_ELE.dmat", TT, false);
    igl::writeDMAT(phantom+"_NODE.dmat", VT, false);
    Vo = Vo.topRows(numVo);
    // BBW - BONE
    Eigen::MatrixXd bc;
    Eigen::VectorXi b;
    if(igl::boundary_conditions(VT, TT, C, Eigen::VectorXi(), BE, Eigen::MatrixXi(), b, bc))
      std::cout<<"boundary condition was generated for "<< b.rows()<<" vertices." <<std::endl;
    else
    {
      std::cout<<"boundary condition generation failed."<<std::endl;
      return 1;
    }
    // Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(b.rows(),G.maxCoeff()+1);
    // for(int i=0;i<BE.rows();i++) bc.col(G(i))+= bc1.col(i);
    // bc1.resize(0, 0);
    b.conservativeResize(b.rows() + 4);
    b.bottomRows(4) = Eigen::VectorXi::LinSpaced(4, numVo+12, numVo+15);
    bc.conservativeResize(b.rows(), bc.cols());
    bc.bottomRows(4) = Eigen::MatrixXd::Zero(4, bc.cols());
    bc.block(bc.rows()-4, 3, 4, 1) = Eigen::VectorXd::Ones(4);
    igl::BBWData bbw_data;
    bbw_data.active_set_params.max_iter = 10;
    bbw_data.verbosity = 2;
    igl::normalize_row_sums(bc, bc);
    igl::bbw(VT, TT, b, bc, bbw_data, W);
    igl::writeDMAT(phantom+"_b.dmat", W, false);
    // BBW - JOINT
    b.resize(21);
    b << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26;
    b = b.array() + numVo;
    Eigen::VectorXi Gj(b.rows());
    Gj << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14, 15, 16, 17;
    bc = Eigen::MatrixXd::Zero(b.rows(), Gj.maxCoeff()+1);
    for(int i=0;i<b.rows();i++) bc(i, Gj(i)) = 1.;
    igl::bbw(VT, TT, b, bc, bbw_data, Wj);
    igl::writeDMAT(phantom+"_j.dmat", Wj, false);
  }
  std::vector<std::map<int, double>> weightMap;
  for(int i=0;i<W.rows();i++)
  {
    std::map<int, double> w;
    for(int j=0;j<W.cols();j++) w[j] = W(i,j);
    weightMap.push_back(w);
  }
  ///////////////////////////READ DATA//////////////////////////////////
  std::ifstream ifs("../take8.kpts");
  MatrixList RAW;
  Eigen::VectorXi raw(16);
  /////to-be changed: raw data whould be transmitted
  raw<<6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23;
  std::cout<<"reading ../take8.kpts..."<<std::flush;
  for(int f=0;f<213;f++)
  {
    Eigen::MatrixXd C1(24, 4);
    for(int i=0;i<24;i++)
      ifs>>C1(i, 0)>>C1(i, 1)>>C1(i, 2)>>C1(i, 3);
    RAW.push_back(igl::slice(C1, raw, 1));    
  }
  ifs.close();
  std::cout<<"done"<<std::endl;
  /////////////////////////phantom scaling//////////////////////////////
  Eigen::MatrixXi BE_s(12, 2);
  BE_s<<0, 3, 6, 9, 9, 3, 0, 6, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11;
  // CALCULATE NEW INTER-JOINT LENGTH
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(14, 2); // BE_s(12), body height, head height
  std::cout<<"calculating inter-joint lengths..."<<std::flush;
  for(int f=0;f<213;f++)
  {
    for(int i=0;i<BE_s.rows();i++)
    {
      if(RAW[f](BE_s(i, 0), 3)<0.6 || RAW[f](BE_s(i, 1), 3)<0.6 ) continue;
      double w = RAW[f](BE_s(i, 0), 3) + RAW[f](BE_s(i, 1), 3);
      double l = (RAW[f].row(BE_s(i,0)) - RAW[f].row(BE_s(i,1))).norm();
      L(i, 0) += w*l;
      L(i, 1) += w;
    }
    if(RAW[f](0, 3)>0.6 &&RAW[f](3, 3)>0.6)
    {
      Eigen::RowVector3d midShoul = (RAW[f].row(0)+RAW[f].row(3))*0.5;
      if(RAW[f](6, 3)>0.6 &&RAW[f](9, 3)>0.6)
      {
        double w = RAW[f](0, 3)+RAW[f](3, 3)+RAW[f](6, 3)+RAW[f](9, 3);
        L(12, 0) += w * (midShoul-(RAW[f].row(6)+RAW[f].row(9))*0.5).norm();
        L(12, 1) += w;
      }
      if(RAW[f](14, 3)>0.3 &&RAW[f](15, 3)>0.3)
      {
        double w = RAW[f](0, 3)+RAW[f](3, 3)+RAW[f](14, 3)+RAW[f](15, 3);
        L(13, 0) += w * (midShoul-(RAW[f].row(14)+RAW[f].row(15))*0.5).norm();
        L(13, 1) += w;
      }
    }    
  }
  L.col(0) = L.col(0).array() * L.col(1).cwiseInverse().array();
  std::cout<<"done"<<std::endl;
  // CALCULATE ORIGINAL TORSO SIZE
  Eigen::VectorXd L0(14); // BE_s(12), body height, head height
  for(int i=0;i<BE_s.rows();i++)
    L0(i) = (C.row(BE_s(i,0)) - C.row(BE_s(i,1))).norm();
  Eigen::RowVector3d midShoul0 = C.row(0)*shoulM + C.row(3)*(1-shoulM);
  L0(12) = (midShoul0 - (C.row(6)+C.row(9))*0.5).norm();
  L0(13) = (midShoul0 - (C.row(14)+C.row(15))*0.5).norm();
  // PRINT
  std::cout<<"<joint lengths>"<<std::endl;
  std::cout<<"body height: "<<L0(12)<<" > "<<L(12, 0)<<" ("<<L(12, 1)<<", "<<(L(12,0)/L0(12)-1)*100.<<"%)"<<std::endl;  
  std::cout<<"head height: "<<L0(13)<<" > "<<L(13, 0)<<" ("<<L(13, 1)<<", "<<(L(13,0)/L0(13)-1)*100.<<"%)"<<std::endl;  
  for(int i=0;i<BE_s.rows();i++)
  {
    std::cout<<BE_s(i, 0)<<"-"<<BE_s(i, 1)<<": "<<L0(i)<<" > "<<L(i, 0)<<" ("<<L(i, 1)<<", "<<(L(i,0)/L0(i)-1)*100.<<"%)"<<std::endl;
    if(L(i,1)<100)
    {
      // std::cout<<"\t-> set as original length"<<std::endl;
      // L(i,0) = L0(i);
      std::cout<<"\t-> set as 0.9"<<std::endl;
      L(i,0) = L0(i)*0.85;
    }
  }
  // CALCULATE NEW JOINT POSITION
  Eigen::MatrixXd C0 = C;
  double h = L(12,0)/L0(12);
  double wB = L(1,0)/L0(1);
  double wT = L(0,0)/L0(0);
  C0.row(9) = C0.row(16)+(C.row(9)-C.row(16))*wB;
  C0.row(6) = C0.row(16)+(C.row(6)-C.row(16))*wB;
  //spine
  for(int i=17;i<21;i++)  
    C0.row(i) = C0.row(i-1)+(C.row(i)-C.row(i-1))*h;
  //head
  for(int i=12;i<16;i++)
    C0.row(i) = C0.row(20) + (C.row(i)-C.row(20));
  Eigen::RowVector3d up=(C.row(19)-C.row(18)).normalized();
  Eigen::RowVector3d right = (C.row(9)-C.row(16)).normalized();
  std::vector<int> shoulLine = {3, 22, 21, 0};
  for(int c:shoulLine)
    C0.row(c) = C0.row(18) + up.dot(C.row(c)-C.row(18))*up*h + right.dot(C.row(c)-C.row(18))*right*wT;
  //limbs
  for(int i=0;i<BE.rows();i++)
  {
    if(BE(i,0)>15) continue;
    int b=0;
    if(BE(i,1)<16)
    {
      for(;b<BE_s.rows();b++) 
        if(BE_s(b,1)==BE(i,1)) break;
    }
    else 
    {
      for(;b<BE_s.rows();b++) 
        if(BE_s(b,1)==BE(i,0)) break;
    }
    C0.row(BE(i,1)) = C0.row(BE(i,0))+L(b,0)/L0(b)*(C.row(BE(i,1))-C.row(BE(i,0)));
  }
  // PHANTOM SCALING
  Eigen::MatrixXd TRANS(18, 3);
  TRANS.topRows(13) = C0.topRows(13)-C.topRows(13);
  Eigen::VectorXi R(5); R<< 19, 23, 24, 25, 26;
  TRANS.bottomRows(5) = igl::slice(C0,R,1) - igl::slice(C,R,1);
  Eigen::MatrixXd VT0 = VT;
  VT = VT + Wj*TRANS;
  Vo = VT.topRows(numVo);
  // CALCULATE NEW INTER-JOINT LENGTHS (for daughter joints)
  L.resize(C.rows(), 1);
  for(int i=0;i<BE.rows();i++)
    L(BE(i,1)) = (C0.row(BE(i,1))-C0.row(BE(i,0))).norm();
  // calculate root pos. and midShoul pos.
  midShoul0 = shoulM * C0.row(0) + (1.-shoulM) * C0.row(3);
  double clavL = (midShoul0 - C0.row(21)).norm();
  double clavR = (midShoul0 - C0.row(22)).norm();
  double shoulL = (midShoul0 - C0.row(0)).norm();
  double shoulR = (midShoul0 - C0.row(3)).norm();
  double mid2spineC = (midShoul0-C0.row(18)).norm();
  // CALCULATE BONE VECTORS
  Eigen::MatrixXd BV(BE.rows(), 3); 
  for(int i=0;i<BE.rows();i++)
    BV.row(i) = (C0.row(BE(i, 1)) - C0.row(BE(i, 0))).normalized();
  // standing height
  double height17 = Vo.col(1).maxCoeff()-C0(17, 1);
  /////////////////////////rotation calculation//////////////////////////////
  // complete R
  Matrix3dList ROT0;
  Eigen::Matrix3d root0;
  root0.col(0) = (C0.row(6)-C0.row(9)).normalized(); // toLhip
  root0.col(1) = BV.row(0); // root-spineN
  root0.col(2) = root0.col(0).cross(root0.col(1));
  Eigen::Matrix3d shoul0;
  shoul0.col(0) = (C0.row(0)-C0.row(3)).normalized(); // toLshoul
  shoul0.col(1) = BV.row(2); // spineC-neck
  shoul0.col(2) = shoul0.col(0).cross(shoul0.col(1));
  Eigen::Matrix3d chest0;
  chest0.col(0) = (root0.col(0) + shoul0.col(0)).normalized();
  chest0.col(1) = BV.row(1); 
  chest0.col(2) = chest0.col(0).cross(chest0.col(1)).normalized();
  chest0.col(0) = chest0.col(1).cross(chest0.col(2));
  Eigen::Matrix3d head0;
  head0.col(0) = (C0.row(13)-C0.row(12)).normalized(); // eye line
  head0.col(1) = ((C0.row(12)+C0.row(13))*0.5-C0.row(20)).normalized();
  head0.col(2) = head0.col(0).cross(head0.col(1)).normalized();
  head0.col(1) = head0.col(2).cross(head0.col(0));
  for(int i=0;i<BE.rows();i++)
  {
    if(BE(i,0)==16) ROT0.push_back(root0);
    else if(BE(i,0)==18 || BE(i,0)==21 || BE(i,0)==22) ROT0.push_back(shoul0);
    else if(BE(i,0)==17) ROT0.push_back(chest0);
    else if(BE(i,0)==19) ROT0.push_back(head0);
    else
    {
      Eigen::Matrix3d axis;
      axis.col(1) = BV.row(i);
      if(i>5 && i<9) axis.col(0) = shoul0.col(0); //left arm
      else if(i>10 && i<14) axis.col(0) = -shoul0.col(0); //right arm
      else if(i>14 && i<18) axis.col(0) = root0.col(0); //left leg
      else if(i>18 && i<22) axis.col(0) = -root0.col(0); //right leg
      axis.col(2) = axis.col(0).cross(axis.col(1)).normalized();
      axis.col(0) = axis.col(1).cross(axis.col(2));
      ROT0.push_back(axis);
    }
  }
  // calculate vQ
  std::vector<RotationList> vQ_vec;
  std::vector<std::vector<Eigen::Vector3d>> vT_vec;
  Eigen::MatrixXd Vv;
  MatrixList C_vec, C_vec1;
  std::vector<int> extBE = {5, 8, 11, 14};
  Eigen::VectorXi legs(9);
  legs<<6, 7, 8, 9, 10, 11, 16, 25, 26;
  for(size_t f=0;f<RAW.size();f++)
  {
    Eigen::MatrixXd C1 = RAW[f];
    C1.conservativeResize(27, 3);
    // C1.row(16) = (1-rootL)*C1.row(6)+rootL*C1.row(9); //root
    C1.row(20) = (C1.row(15)+C1.row(14))*0.5; //head
    Eigen::RowVector3d midShoul = C1.row(0)*shoulM + C1.row(3)*(1.-shoulM);
    Eigen::RowVector3d toLshoul = (C1.row(0)-C1.row(3)).normalized();
    C1.row(21) = midShoul + toLshoul * clavL; //clavL
    C1.row(0) = midShoul + toLshoul * shoulL; //shoulL
    C1.row(22) = midShoul - toLshoul * clavR; //clavR
    C1.row(3) = midShoul - toLshoul * shoulR; //shoulR
    Eigen::RowVector3d toSpineC = (Eigen::RowVector3d::UnitZ().cross(toLshoul)).cross(toLshoul).normalized();
    C1.row(18) = midShoul+toSpineC*mid2spineC;
    C1.row(19) = C1.row(18)-toSpineC*L(19);
    Eigen::RowVector3d toSpineN = (toSpineC - Eigen::RowVector3d::UnitZ()).normalized();
    C1.row(17) = C1.row(18)+L(18)*toSpineN;
    Eigen::RowVector3d toLhip = C1.row(6)-C1.row(9);
    toLhip(2) = 0; toLhip.normalize();
    if(C1(17, 2)<height17)
    {
      double trans = height17 - C1(17, 2); //up
      C1.row(18) = midShoul - toSpineC*(midShoul(2)-C1(18, 2)-trans*0.75) / toSpineC(2);
      C1.row(17) = C1.row(18) - toSpineN * (C1(18, 2)-height17)/toSpineN(2); // spineN

      Eigen::RowVector3d back =toLhip.cross(toSpineC);
      trans = std::sqrt(mid2spineC*mid2spineC-(midShoul-C1.row(18)).squaredNorm());
      C1.row(18) = C1.row(18) + back*trans;
      C1.row(19) = C1.row(18) + (midShoul-C1.row(18))/mid2spineC*L(19);
      C1.row(17) = C1.row(17) + back*trans;
      // back =toLhip.cross(toSpineN); --> to be accurate
      trans = std::sqrt(L(18)*L(18)-(C1.row(18)-C1.row(17)).squaredNorm());
      C1.row(17) = C1.row(17) + back*trans;
    }
    C1.row(16) = C1.row(17) - Eigen::RowVector3d::UnitZ()*L(17);
    C1.row(9) =  C1.row(16) - toLhip*L(9);
    C1.row(6) =  C1.row(16) + toLhip*L(6);
    
    for(int i=0;i<BE.rows();i++)
    {
      if(BE(i,1)>22) C1.row(BE(i,1)) = C1.row(BE(i,0)) + (C1.row(BE(i-1,1))-C1.row(BE(i-1,0))).normalized()*L(BE(i,1)); // extremities
      else if(BE(i,0)<16) C1.row(BE(i,1)) = C1.row(BE(i,0)) + (C1.row(BE(i,1))-C1.row(BE(i,0))).normalized()*L(BE(i,1)); // limbs
    }

    C_vec.push_back(C1);

    //calculate rotation
    RotationList vQ(BE.rows());
    std::vector<Eigen::Vector3d> vT(BE.rows());
    // complete R
    Eigen::Matrix3d root1;
    root1.col(0) = toLhip; // toLhip
    root1.col(1) = (C1.row(17)-C1.row(16)).normalized(); // root-spineN
    root1.col(2) = root1.col(0).cross(root1.col(1));
    Eigen::Matrix3d shoul1;
    shoul1.col(0) = toLshoul; // toLshoul
    shoul1.col(1) = (C1.row(19)-C1.row(18)).normalized(); // spineC-neck
    shoul1.col(2) = shoul1.col(0).cross(shoul1.col(1));
    Eigen::Matrix3d chest1;
    chest1.col(0) = (toLhip + toLshoul).normalized(); // 
    chest1.col(1) = (C1.row(18)-C1.row(17)).normalized(); 
    chest1.col(2) = chest1.col(0).cross(chest1.col(1)).normalized();
    chest1.col(0) = chest1.col(1).cross(chest1.col(2));
    Eigen::Matrix3d head1;
    head1.col(0) = (C1.row(13)-C1.row(12)).normalized(); // eye line
    head1.col(1) = ((C1.row(12)+C1.row(13))*0.5-C1.row(20)).normalized();
    head1.col(2) = head1.col(0).cross(head1.col(1)).normalized();
    head1.col(1) = head1.col(2).cross(head1.col(0));
    Eigen::Quaterniond rootQ(root1*root0.inverse());
    Eigen::Quaterniond shoulQ(shoul1*shoul0.inverse());
    Eigen::Quaterniond chestQ(chest1*chest0.inverse());
    Eigen::Quaterniond headQ(head1*head0.inverse());

    double thetaLH, thetaRH;    
    for(int i=0;i<BE.rows();i++)
    {
      if(BE(i,0)==16) vQ[i] = rootQ;
      else if(BE(i,0)==18 || BE(i,0)==21 || BE(i,0)==22) vQ[i] = shoulQ;
      else if(BE(i,0)==17) vQ[i] = chestQ;
      else if(BE(i,0)==19) vQ[i] = headQ;
      /////////leg fix///////////
      else if(i>14) vQ[i] = rootQ;
      ///////////////////////////
      else
      {
        Eigen::Matrix3d axis;
        axis.col(1) = (C1.row(BE(i,1))-C1.row(BE(i,0))).normalized();
        if(i>5 && i<9) axis.col(0) = shoul1.col(0); //left arm
        else if(i>10 && i<14) axis.col(0) = -shoul1.col(0); //right arm
        else if(i>14 && i<18) axis.col(0) = root1.col(0); //left leg
        else if(i>18 && i<22) axis.col(0) = -root1.col(0); //right leg
        axis.col(2) = axis.col(0).cross(axis.col(1)).normalized();
        axis.col(0) = axis.col(1).cross(axis.col(2));
        
        if(i==8 || i==13) //hands
        {
          Eigen::Matrix3d axis1 = axis;
          axis1.col(0) = Eigen::Vector3d::UnitZ();
          axis1.col(2) = axis1.col(0).cross(axis1.col(1)).normalized();
          axis1.col(0) = axis1.col(1).cross(axis1.col(2));
          Eigen::AngleAxisd q(axis1*axis.inverse());
          axis = axis1;
          if(i==8)
          {
            thetaLH = q.angle()/3.;
            if(axis.col(1).dot(q.axis())<0) thetaLH = -thetaLH;
          }
          if(i==13) 
          {
            thetaRH = q.angle()/3.;
            if(axis.col(1).dot(q.axis())<0) thetaRH = -thetaRH;
          }
        }
        vQ[i] = axis*ROT0[i].inverse();
      }
    }
    vQ[6] = Eigen::AngleAxisd(thetaLH, (C1.row(BE(6,1))-C1.row(BE(6,0))).normalized()) * vQ[6];
    vQ[7] = Eigen::AngleAxisd(thetaLH*2, (C1.row(BE(7,1))-C1.row(BE(7,0))).normalized()) * vQ[7];
    vQ[11] = Eigen::AngleAxisd(thetaRH, (C1.row(BE(11,1))-C1.row(BE(11,0))).normalized()) * vQ[11];
    vQ[12] = Eigen::AngleAxisd(thetaRH*2, (C1.row(BE(12,1))-C1.row(BE(12,0))).normalized()) * vQ[12];

    //set new joint points
    for(int i=0;i<BE.rows();i++)
      C1.row(BE(i, 1)) = C1.row(BE(i, 0)) + (vQ[i]*BV.row(i).transpose()*L(BE(i,1))).transpose();
    for(int i=12;i<16;i++) C1.row(i) = C1.row(15) + (headQ*(C0.row(i)-C0.row(15)).transpose()).transpose();
    C_vec1.push_back(C1);
    // Eigen::RowVector3d toRhip = (C1.row(9)-C1.row(6)).normalized();
    for(int i=0;i<BE.rows();i++) vT[i] = C1.row(BE(i,0)).transpose()-vQ[i]*C0.row(BE(i,0)).transpose();
    vQ_vec.push_back(vQ);
    vT_vec.push_back(vT);

    Vv.conservativeResize(Vv.rows() + C1.rows(), 3);
    Vv.bottomRows(C1.rows()) = C1;
  }

  std::ifstream ifsNODE(phantom+".node");
  int nodeNum, tmp;
  ifsNODE>>nodeNum>>tmp>>tmp>>tmp;
  Eigen::MatrixXd NODE(nodeNum,3);
  for(int i=0;i<nodeNum;i++)
    ifsNODE>>tmp>>NODE(i,0)>>NODE(i,1)>>NODE(i,2);
  ifsNODE.close();
  // NODE = NODE.rowwise() + (VT.colwise().maxCoeff() - NODE.colwise().maxCoeff());
  std::cout<<NODE.colwise().maxCoeff()<<std::endl<<NODE.colwise().minCoeff()<<std::endl;
  std::cout<<Vo.colwise().maxCoeff()<<std::endl<<Vo.colwise().minCoeff()<<std::endl;
  Eigen::SparseMatrix<double> bary = GenBary(NODE, VT0, TT);

  igl::opengl::glfw::Viewer vr;
  Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
  Eigen::MatrixXi Ff;
  vr.data().set_edges(C0, BE, sea_green);
  vr.data().set_mesh(Vo, Fo);
  vr.data().set_points(C0, sea_green);
  vr.data().point_size = 8;
  vr.data().show_overlay_depth = false;
  int frame(0);
  vr.callback_key_down = [&](igl::opengl::glfw::Viewer _vr, unsigned int key, int modifiers)->bool{
  switch (key)
  {
  case ']':
    frame = std::min((int)C_vec.size()-1, frame+1);
    vr.data().set_edges(C_vec1[frame], BE, sea_green);
    vr.data().set_points(C_vec[frame], sea_green);
    break;
  case '[':
    frame = std::max(0, frame-1);
    vr.data().set_edges(C_vec1[frame], BE, sea_green);
    vr.data().set_points(C_vec[frame], sea_green);
    break;
  case 'p':
  case 'P':
    {
      Eigen::MatrixXd U;
      myDqs(VT, weightMap, vQ_vec[frame], vT_vec[frame], U);
      Eigen::MatrixXd NODE1 = bary*U;
      std::ofstream ofs(phantom+"_"+std::to_string(frame)+".node");
      ofs<<NODE.rows()<<"   3   0   0"<<std::endl;
      for(int i=0;i<NODE.rows();i++)
      {
        ofs<<i<<" "<<NODE1.row(i)<<std::endl;
        std::cout<<"\rprinting..."+std::to_string(i)+"/"+std::to_string(NODE.rows())<<std::flush;
      }
      ofs.close();
      std::cout<<"\rprinting...done ("+phantom+"_"+std::to_string(frame)+".node)"<<std::endl;
    }
    break;
  default:
    break;
  }
  {
    Eigen::MatrixXd U;
    myDqs(VT, weightMap, vQ_vec[frame], vT_vec[frame], U);
    
    vr.data().set_vertices(U.topRows(numVo));
    vr.data().compute_normals();
  }
  return true;};
  vr.launch();

  return 0;
}
  
  