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
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/boundary_conditions.h>
#include <igl/bbw.h>
#include <Eigen/Dense>

void PrintUsage(){
  std::cout<<"./poseEst [phantom name]"<<std::endl;
}
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>>
    RotationList;
typedef std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
    MatrixList;
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
  Eigen::RowVector3d midShoul0 = (C.row(0)+C.row(3))*0.5;
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
      std::cout<<"\t-> set as original length"<<std::endl;
      L(i,0) = L0(i);
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
    if(BE(i,1)<16) for(;b<BE_s.rows();b++) if(BE_s(b,1)==BE(i,1)) break;
    else for(;b<BE_s.rows();b++) {if(BE_s(b,1)==BE(i,0)) break;}
    C0.row(BE(i,1)) = C0.row(BE(i,0))+L(b,0)/L0(b)*(C.row(BE(i,1))-C.row(BE(i,0)));
  }
  // PHANTOM SCALING
  Eigen::MatrixXd TRANS(18, 3);
  TRANS.topRows(13) = C0.topRows(13)-C.topRows(13);
  TRANS.bottomRows(5) = C0.middleRows(13, 5) - C.middleRows(13, 5);
  VT = VT + Wj*TRANS;
  Vo = VT.topRows(numVo);
  // CALCULATE NEW INTER-JOINT LENGTHS (for daughter joints)
  L.resize(C.rows(), 1);
  for(int i=0;i<BE.rows();i++)
    L(BE(i,1)) = (C0.row(BE(i,1))-C0.row(BE(i,0))).norm();
  // calculate root pos. and midShoul pos.
  double rootL = (C0.row(16)-C0.row(9)).norm() / (C0.row(6)-C0.row(9)).norm();
  double shoulM = 1.-(C0.row(19)-C0.row(0)).dot((C0.row(3)-C0.row(0)).normalized()) / (C0.row(3)-C0.row(0)).norm();
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
  
  /////////////////////////rotation calculation//////////////////////////////
  // complete R
  Eigen::Matrix3d root0;
  root0.col(0) = (C.row(17)-C.row(16)).normalized(); // root-spineN
  root0.col(1) = (C.row(9)-C.row(6)).normalized(); // toRhip
  root0.col(2) = root0.col(0).cross(root0.col(1));
  Eigen::Matrix3d shoul0;
  shoul0.col(0) = (C.row(19)-C.row(18)).normalized(); // spineC-neck
  shoul0.col(1) = (C.row(3)-C.row(0)).normalized(); // toRshoul
  shoul0.col(2) = shoul0.col(0).cross(shoul0.col(1));
  Eigen::Matrix3d head0;
  head0.col(0) = (C.row(13)-C.row(12)).normalized(); // eye line
  head0.col(1) = ((C.row(12)+C.row(13))*0.5-C.row(20)).normalized();
  head0.col(2) = head0.col(0).cross(head0.col(1));
  // calculate vQ
  std::vector<RotationList> vQ_vec;
  std::vector<std::vector<Eigen::Vector3d>> T_vec;
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
    Eigen::RowVector3d midShoul = C1.row(0)*shoulM + C1.row(3)*(1-shoulM);
    Eigen::RowVector3d toLshoul = (C1.row(0)-C1.row(3)).normalized();
    C1.row(21) = midShoul + toLshoul * clavL; //clavL
    C1.row(0) = midShoul + toLshoul * shoulL; //shoulL
    C1.row(22) = midShoul - toLshoul * clavR; //clavR
    C1.row(3) = midShoul - toLshoul * shoulR; //shoulR
    Eigen::RowVector3d toSpineC = (Eigen::RowVector3d::UnitZ().cross(toLshoul)).cross(toLshoul).normalized();
    C1.row(18) = midShoul+toSpineC*mid2spineC;
    C1.row(19) = C1.row(18)-toSpineC*L(19);
    C1.row(17) = C1.row(18)+L(18)*(toSpineC - Eigen::RowVector3d::UnitZ()).normalized();
    Eigen::RowVector3d toLhip = C1.row(6)-C1.row(9);
    toLhip(2) = 0; toLhip.normalize();
    C1.row(16) = C1.row(17) - Eigen::RowVector3d::UnitZ()*L(17);
    C1.row(9) =  C1.row(16) - toLhip*L(9);
    C1.row(6) =  C1.row(16) + toLhip*L(6);
    
    // Eigen::RowVector3d toSpineC = toLshoul.cross(Eigen::RowVector3d(C1.row(16)-midShoul).cross(toLshoul)).normalized();
    for(int i=0;i<BE.rows();i++)
    {
      if(BE(i,1)>22) C1.row(BE(i,1)) = C1.row(BE(i,0)) + (C1.row(BE(i-1,1))-C1.row(BE(i-1,0))).normalized()*L(BE(i,1)); // extremities
      else if(BE(i,0)<16) C1.row(BE(i,1)) = C1.row(BE(i,0)) + (C1.row(BE(i,1))-C1.row(BE(i,0))).normalized()*L(BE(i,1)); // limbs
      // else if (i<3) C1.row(BE(i,1)) = C1.row(BE(i,0)) + (midShoul-C1.row(16)).normalized()*L(BE(i,1)); // spine
    }

    C_vec.push_back(C1);

    //calculate rotation
    RotationList vQ(BE.rows());
    // complete R
    Eigen::Matrix3d root1;
    root1.col(0) = (C1.row(17)-C1.row(16)).normalized(); // root-spineN
    root1.col(1) = (C1.row(9)-C1.row(6)).normalized(); // toRhip
    root1.col(2) = root1.col(0).cross(root1.col(1));
    Eigen::Quaterniond rootQ(root1*root0.inverse());
    Eigen::Matrix3d shoul1;
    shoul1.col(0) = (C1.row(19)-C1.row(18)).normalized(); // spineC-neck
    shoul1.col(1) = (C1.row(3)-C1.row(0)).normalized(); // toRshoul
    shoul1.col(2) = shoul1.col(0).cross(shoul1.col(1));
    Eigen::Quaterniond shoulQ(shoul1*shoul0.inverse());
    Eigen::Matrix3d head1;
    head1.col(0) = (C1.row(13)-C1.row(12)).normalized(); // eye line
    head1.col(1) = ((C1.row(12)+C1.row(13))*0.5-C1.row(20)).normalized();
    head1.col(2) = head1.col(0).cross(head1.col(1));
    Eigen::Quaterniond headQ(head1*head0.inverse());
    
    for(int i=0;i<BE.rows();i++)
    {
      if(BE(i,0)==16) vQ[i] = rootQ;
      else if(BE(i,0)==18) vQ[i] = shoulQ;
      else if(BE(i,0)==19) vQ[i] = headQ;
      else if(i>=14 && i<=21) vQ[i] = rootQ;
      else
      {
        Eigen::Vector3d v1 = (C1.row(BE(i, 1))-C1.row(BE(i, 0))).normalized();
        vQ[i].setFromTwoVectors(BV.row(i), v1);
      }
    }

    //set new joint points
    
    for(int i=0;i<BE.rows();i++)
      C1.row(BE(i, 1)) = C1.row(BE(i, 0)) + (vQ[i]*BV.row(i).transpose()*L(BE(i,1))).transpose();
    for(int i=12;i<16;i++) C1.row(i) = C1.row(15) + (headQ*(C0.row(i)-C0.row(15)).transpose()).transpose();
    C_vec1.push_back(C1);
    // Eigen::RowVector3d toRhip = (C1.row(9)-C1.row(6)).normalized();

    Vv.conservativeResize(Vv.rows() + C1.rows(), 3);
    Vv.bottomRows(C1.rows()) = C1;
  }
  //   Eigen::RowVector3d midShoul = (C1.row(0)+C1.row(3))*0.5;
  //   Eigen::RowVector3d cloesestSpineNV = (toRhip.cross((midShoul-C1.row(6)).cross(toRhip))).normalized();
  //   C1.row(17) = C1.row(16) + L(17) * cloesestSpineNV;
  //   ((C1.row(6)+C1.row(10))*0.5-C).cross(C1.row(9)-C1.row(6))
  //   // calculate complete R
  //   Eigen::Matrix3d root1;
  //   root1.col(0) = (C1.row(1)-C1.row(0)).normalized();
  //   root1.col(1) = (C1.row(13)-C1.row(0)).normalized(); // left hip
  //   root1.col(2) = root1.col(0).cross(root1.col(1));
  //   Eigen::Matrix3d shoul1;
  //   shoul1.col(0) = (C1.row(2)-C1.row(3)).normalized();
  //   shoul1.col(1) = (C1.row(6)-C1.row(10)).normalized(); // shoulder line
  //   shoul1.col(2) = shoul1.col(0).cross(shoul1.col(1));
  //   Eigen::Matrix3d head1;
  //   head1.col(0) = (C1.row(14)-C1.row(15)).normalized(); // eye line
  //   head1.col(1) = head0.col(0).cross(C1.row(14)-C1.row(20)).normalized(); // eye line
  //   head1.col(2) = head1.col(0).cross(head1.col(1));
  //   //calculate vQ
  //   RotationList vQ(16);
    

  // // }

  igl::opengl::glfw::Viewer vr;
  Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
  Eigen::MatrixXi Ff;
  vr.data().set_edges(C0, BE, sea_green);
  vr.data().set_mesh(Vv, Ff);
  vr.data().set_points(C0, sea_green);
  vr.data().point_size = 8;
  int frame(0);
  vr.callback_key_down = [&](igl::opengl::glfw::Viewer _vr, unsigned int key, int modifiers)->bool{
  switch (key)
  {
  case ']':
    frame = std::min((int)C_vec.size()-1, frame+1);
    vr.data().set_edges(C_vec[frame], BE, sea_green);
    vr.data().set_points(C_vec1[frame], sea_green);
    break;
  case '[':
    frame = std::max(0, frame-1);
    vr.data().set_edges(C_vec[frame], BE, sea_green);
    vr.data().set_points(C_vec1[frame], sea_green);
    break;
  default:
    break;
  }return true;};
  vr.launch();
  // vr.data().set_mesh(Vo,Fo);
  // vr.data().show_overlay_depth = false;
  // vr.data().show_lines = false;
  // vr.data().point_size = 8;
  // vr.data().set_points(C0, Eigen::RowVector3d(1., 0., 0.));
  // vr.append_mesh();
  // vr.data().set_edges(C, BE, sea_green);
  // vr.data().set_points(C, sea_green);
  // vr.data().point_size = 8;
  // vr.data().show_overlay_depth = false;
  // vr.launch();

  return 0;
}
  
  
  /*
  igl::readTGF("../yolo.tgf", C, BE);
  double spineR1 = (C.row(1)-C.row(0)).norm()/(C.row(3)-C.row(0)).norm();
  double spineR2 = (C.row(2)-C.row(1)).norm()/(C.row(3)-C.row(0)).norm();
  double spineR3 = 1. - spineR1 - spineR2;

  ////////////////////
  Eigen::Matrix3d rootM;
  rootM.col(0) = (C.row(1)-C.row(0)).normalized();
  rootM.col(1) = (C.row(13)-C.row(0)).normalized(); // left hip
  rootM.col(2) = rootM.col(0).cross(rootM.col(1));
  Eigen::Matrix3d shoulM;
  shoulM.col(0) = (C.row(2)-C.row(3)).normalized();
  shoulM.col(1) = (C.row(6)-C.row(10)).normalized(); // shoulder line
  shoulM.col(2) = shoulM.col(0).cross(shoulM.col(1));
  ////////////////////
  std::ifstream ifs("../take8.kpts");
  MatrixList C_vec;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<int> setJ = {0, 1, 2, 3, 4, 5, 9};
  Eigen::VectorXi raw(C.rows() - setJ.size());
  for(int i, j=0;i<C.rows();i++)
  {
    if(i==setJ[j]) {j++; continue;}
    raw(i-j) = i;
  }
  Eigen::VectorXd L=Eigen::VectorXd::Zero(BE.rows());
  for(int f=0;f<213;f++)
  {
    Eigen::MatrixXd C1(C);
    Eigen::VectorXd conf(C.rows());
    for(int i=0;i<C.rows();i++)
      ifs>>C1(i, 0)>>C1(i, 1)>>C1(i, 2)>>conf(i);
    C1.row(0) = 0.5*(C1.row(13)+C1.row(16)); //root
    C1.row(3) = 0.5*(C1.row(6)+C1.row(10)); //neck
    C1.row(4) = 0.5*(C1.row(22)+C1.row(23)); //head

    for(int i=3;i<BE.rows();i++)
    {
      if(conf(BE(i, 0))>0.6 && conf(BE(i, 1))>0.6)
      {
        W(i) += conf(BE(i, 0))+conf(BE(i, 1));
        L(i) += (C1.row(BE(i, 0))-C1.row(BE(i, 1))).norm() * (conf(BE(i, 0))+conf(BE(i, 1)));
      }
    }
    if(conf(0)>0.6 && conf(3)>0.6)
    {
      double w = conf(0)+conf(3);
      double l = (C1.row(3)-C1.row(0)).norm();
      W(0) += w; W(1) += w; W(2) += w;
      L(0) += l*w*spineR1; L(1) += l*w*spineR2; L(2) += l*w*spineR3;
    }
    C_vec.push_back(C1);
  }
  ifs.close();
  Eigen::VectorXd L2 = L;
  for(int i=0;i<BE.rows();i++)
  {
    if(W(i)>100) L(i) = L(i) / W(i);
    else L(i) = (C.row(BE(i,0))-C.row(BE(i,1))).norm();
    L2(i) = L(i) * L(i);
    C.row(BE(i, 1)) = C.row(BE(i, 0)) + (C.row(BE(i, 1))-C.row(BE(i, 0))).normalized()*L(i);
    std::cout<<i<<" ("<<W(i)<<"): "<<L(i)<<std::endl;
  }

  double rootSpine2 = (C.row(1)-C.row(0)).squaredNorm(); //L2()
  double rootL = std::sqrt(rootSpine2);
  double neckToSpine2 = (L(1) + L(2)) * (L(1) + L(2));
  double alpha = 0.8;
  MatrixList C_vec1;
  std::vector<RotationList> vQ_vec;
  std::vector<Eigen::RowVector3d> T_vec;
  for(Eigen::MatrixXd &C1:C_vec)
  {
    Eigen::RowVector3d toRoot = C1.row(0) - C1.row(3);
    Eigen::RowVector3d toLeftShoul = C1.row(6) - C1.row(3);
    Eigen::RowVector3d toLeftHip = C1.row(13) - C1.row(0);
    double d2 = toRoot.squaredNorm();
    double x2 = (neckToSpine2 - L2(0) + d2)*0.5;
    x2 = (x2*x2)/d2;
    if(neckToSpine2 - x2>0)
    {
      double h = std::sqrt(neckToSpine2 - x2);
      Eigen::RowVector3d rootDir = toLeftShoul.cross(toRoot).normalized()*h*alpha-toRoot.normalized()*std::sqrt(x2);
      C1.row(1) = C1.row(0) + toLeftHip.cross(rootDir).cross(toLeftHip).normalized()*L(0);
      d2 = (C1.row(3)-C1.row(1)).squaredNorm();
      double midSpine2 = (C1.row(1)-C1.row(2)).squaredNorm();
      double x = (midSpine2 - (C1.row(2)-C1.row(3)).squaredNorm() + d2)*0.5 / std::sqrt(d2);
      h = std::sqrt(midSpine2 - x*x);
      Eigen::RowVector3d toSpineL = C1.row(1)-C1.row(3);
      C1.row(2) = C1.row(1) + (C1.row(3)-C1.row(1)).normalized() * x + toLeftShoul.cross(toSpineL).normalized() * h;
    }
    else 
    {
      C1.row(1) = C1.row(0) + toLeftHip.cross(Eigen::RowVector3d(C1.row(3)-C1.row(0))).cross(toLeftHip).normalized()*L(0);
      C1.row(2) = (C1.row(1) + C1.row(3))*0.5;
    }
////////////////////quaternion
    RotationList vQ(BE.rows());
    Eigen::Matrix3d rootM1;
    rootM1.col(0) = (C1.row(1)-C1.row(0)).normalized();
    rootM1.col(1) = (C1.row(13)-C1.row(0)).normalized(); // left hip
    rootM1.col(2) = rootM1.col(0).cross(rootM1.col(1));
    Eigen::Quaterniond rootR = Eigen::Quaterniond(rootM1*rootM.inverse());
    Eigen::Matrix3d shoulM1;
    shoulM1.col(0) = (C1.row(2)-C1.row(3)).normalized();
    shoulM1.col(1) = (C1.row(6)-C1.row(10)).normalized(); // shoulder line
    shoulM1.col(2) = shoulM1.col(0).cross(shoulM1.col(1));
    Eigen::Quaterniond shoulR = Eigen::Quaterniond(shoulM1*shoulM.inverse());
    Eigen::MatrixXd C2 = C1;
    for(int j=0;j<BE.rows();j++)
    {
      Eigen::Vector3d v0 = (C.row(BE(j, 1))-C.row(BE(j, 0))).normalized();
      if(BE(j, 0)==0) vQ[j]=rootR;
      else if(BE(j, 0)==2 || BE(j, 0)==5 || BE(j, 0)==9) vQ[j]=shoulR;
      else
      {
        Eigen::Vector3d v1 = (C1.row(BE(j, 1))-C1.row(BE(j, 0))).normalized();
        vQ[j].setFromTwoVectors(v0, v1);
      }
      /////////////////////set points
      C2.row(BE(j, 1)) = C2.row(BE(j,0))+(vQ[j]*v0*L(j)).transpose();
    }
    Eigen::RowVector3d trans = (C1.row(6)-C2.row(6)+C1.row(10)-C2.row(10))*0.5;
    C2 = C2.rowwise() + trans;
    T_vec.push_back(C2.row(0));
    vQ_vec.push_back(vQ);
    C_vec1.push_back(C2);
    V.conservativeResize(V.rows() + C1.rows(), 3);
    V.bottomRows(C1.rows()) = C1;
  }
  ifs.close();

  // for(int f=0;f<C_vec.size();f++)
  // {
  //   Eigen::MatrixXd C1 = C_vec[f];
  //   for(int i=0;i<BE.rows();i++)
  //     C1.row(BE(i, 1)) = C1.row(BE(i, 0)) + (C_vec[f].row(BE(i, 1))-C_vec[f].row(BE(i, 0))).normalized()*L(i);
  //   C_vec1.push_back(C1);
  // }

  igl::opengl::glfw::Viewer vr;
  Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
  vr.data().set_mesh(V, F);
  vr.data().set_edges(C, BE, sea_green);
  Eigen::MatrixXd Color = Eigen::RowVector3d(1., 0., 0.).replicate(raw.rows(), 1);
  // for(int j:setJ) Color.row(j) = Eigen::RowVector3d(1., 0., 0.);
  vr.data().set_points(C, Color);
  vr.data().point_size = 8;
  int frame(0);
  vr.callback_pre_draw = [&](igl::opengl::glfw::Viewer _vr)->bool{
    if(vr.core().is_animating)
    {
      vr.data().set_edges(C_vec1[frame], BE, sea_green);
      vr.data().set_points(igl::slice(C_vec[frame++], raw, 1), Color);
      if(frame == C_vec.size()-1)
      {
        frame = 0;
        vr.core().is_animating = false;
      }
    }
    return false;
  };
  vr.callback_key_down = [&](igl::opengl::glfw::Viewer _vr, unsigned int key, int modifiers)->bool{
    switch (key)
    {
    case ' ':
      vr.core().is_animating = !vr.core().is_animating;
      break;
    case ']':
      frame = std::min((int)C_vec.size()-1, frame+1);
      vr.data().set_edges(C_vec1[frame], BE, sea_green);
      vr.data().set_points(igl::slice(C_vec[frame], raw, 1), Color);
      break;
    case '[':
      frame = std::max(0, frame-1);
      vr.data().set_edges(C_vec1[frame], BE, sea_green);
      vr.data().set_points(igl::slice(C_vec[frame], raw, 1), Color);
      break;
    case 'p':
    case 'P':
    {
      Eigen::MatrixXd posture(BE.rows() + 1, 4);
      for (int i = 0; i < BE.rows(); i++)
        posture.row(i) = vQ_vec[frame][i].coeffs().transpose();
      posture.bottomLeftCorner(1, 3) = T_vec[frame];
      std::string fname = igl::file_dialog_save();
      igl::writeDMAT(fname, posture);      
    }
      break;
    default:
      break;
    }
    return true;
  };
  vr.launch();
  return 0;
}
*/