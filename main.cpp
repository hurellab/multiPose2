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
#include <igl/readPLY.h>
#include <igl/dqs.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/boundary_conditions.h>
#include <igl/directed_edge_parents.h>
#include <igl/in_element.h>
#include <igl/bbw.h>
#include <Eigen/Dense>

// #define ORIGINAL //option to see the original data --> uncomment to see the original data

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

void PrintUsage(){
  std::cout<<"./poseEst [phantom name] [posture data]"<<std::endl;
}
int main(int argc, char *argv[])
{
  if(argc < 3)
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

  // read SHELL
  Eigen::MatrixXd Vo;
  Eigen::MatrixXi Fo;
  igl::readPLY(phantom+".ply", Vo, Fo);
  int numVo = Vo.rows();
  Eigen::MatrixXd Wo;

  if(!igl::readDMAT(phantom+"_Wo.dmat", Wo))
  {
    Eigen::MatrixXd VT;
    Eigen::MatrixXi TT;
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

    igl::BBWData bbw_data;
    bbw_data.active_set_params.max_iter = 10;
    bbw_data.verbosity = 2;
    igl::normalize_row_sums(bc, bc);
    igl::bbw(VT, TT, b, bc, bbw_data, Wo);
    Wo = Wo.topRows(Vo.rows());
    igl::writeDMAT(phantom+"_Wo.dmat", Wo, false);
  }
  std::vector<std::map<int, double>> weightMap;
  for(int i=0;i<Wo.rows();i++)
  {
    std::map<int, double> w;
    for(int j=0;j<Wo.cols();j++) w[j] = Wo(i,j);
    weightMap.push_back(w);
  }
  //////////////////// skip phantom scaling /////////////////////////////
  //////////////////// read data ////////////////////////////////////////
  std::ifstream ifs(argv[2]);
  if(!ifs.is_open())
  {
    std::cout<<argv[2]<<" is not open"<<std::endl;
    return 1;
  }
  std::vector<std::map<int, Eigen::MatrixXd>> frameData;
  std::map<int, Eigen::MatrixXd> aFrame;
  int maxNum(0);
  std::map<int, int> nanCount;
  while(!ifs.eof())
  {
    std::string aLine, key;
    getline(ifs, aLine);
    std::stringstream ss(aLine);
    ss>>key;
    int num(0);
    if(key=="f")
    {
      ss >> num;
      if(num>0) frameData.push_back(aFrame);
      aFrame.clear();
    }
    else if(key=="p")
    {
      double x, y;
      ss >> num >> x >> y;
      if (num>0) //if there is a posture data
      {
        Eigen::MatrixXd C1 = Eigen::MatrixXd::Constant(35, 3, -__DBL_MAX__);
        std::vector<bool> det(34, false);
        for(int i=0;i<34;i++)
        {
          getline(ifs, aLine);
          if(aLine.find("nan")!=std::string::npos){
            nanCount[i]++;
            continue;
          }
          det[i] = true;
          std::stringstream ss1(aLine);
          ss1>>C1(i, 0)>>C1(i, 1)>>C1(i, 2);
        }
        C1.row(34)<<x, y, 0;
        aFrame[num] = C1;
      }
      else
      {
        Eigen::MatrixXd C1(1, 2);
        C1<<x, y;
        aFrame[-num] = C1;
      }
    }
  }
  std::cout<<"<not detected joints>"<<std::endl;
  for(int i=0;i<34;i++) std::cout<<i<<"\t"<<nanCount[i]<<std::endl;
  //////////////////// change to posture data ////////////////////////////
  //1. default rotation
  // CALCULATE BONE VECTORS
  Eigen::MatrixXd BV(BE.rows(), 3); 
  for(int i=0;i<BE.rows();i++)
    BV.row(i) = (C.row(BE(i, 1)) - C.row(BE(i, 0))).normalized();
  // SET SOME EXCEPTIONAL ROTATION MATRICES
  Eigen::Matrix3d root0;
  root0.col(0) = (C.row(12)-C.row(0)).normalized(); // toLhip
  root0.col(2) = BV.row(1); // root-spineN
  root0.col(1) = root0.col(2).cross(root0.col(0)).normalized();
  root0.col(0) = root0.col(1).cross(root0.col(2)).normalized();
  Eigen::Matrix3d shoul0;
  shoul0.col(0) = (C.row(3)-C.row(8)).normalized(); // toLshoul
  shoul0.col(2) = BV.row(3); // spineC-neck
  shoul0.col(1) = shoul0.col(2).cross(shoul0.col(0)).normalized();
  shoul0.col(0) = shoul0.col(1).cross(shoul0.col(2)).normalized();
  Eigen::Matrix3d chest0;
  chest0.col(0) = (root0.col(0) + shoul0.col(0)).normalized();
  chest0.col(2) = BV.row(2); 
  chest0.col(1) = chest0.col(2).cross(chest0.col(0)).normalized();
  chest0.col(0) = chest0.col(1).cross(chest0.col(2)).normalized();
  Eigen::Matrix3d head0;
  head0.col(0) = root0.col(0); // eye line
  head0.col(2) = (C.row(22)-C.row(7)-Eigen::RowVector3d::UnitY()*8.).normalized(); //adjusted
  head0.col(1) = head0.col(2).cross(head0.col(0)).normalized();
  head0.col(2) = head0.col(0).cross(head0.col(1)).normalized();

  // SET THE DEFAULT ROTATION MATRICES FOR ALL THE JOINTS
  Matrix3dList ROT0;
  for(int i=0;i<BE.rows();i++)
  {
    if(BE(i,0)==0) ROT0.push_back(root0);
    else if(BE(i,0)==2 || BE(i,0)==3 || BE(i,0)==8) ROT0.push_back(shoul0);
    else if(BE(i,0)==1) ROT0.push_back(chest0);
    else if(BE(i,0)==7) ROT0.push_back(head0);
    else
    {
      Eigen::Matrix3d axis;
      if(i==17||i==21){
        axis.col(1) = (Eigen::Vector3d::UnitZ()-Eigen::Vector3d(BV.row(i))).normalized();
        axis.col(2) = root0.col(2);
        axis.col(0) = axis.col(1).cross(axis.col(2)).normalized();
        axis.col(2) = axis.col(0).cross(axis.col(1)).normalized();
      }
      else{
        axis.col(2) = -BV.row(i);
        if(i==15||i==16||i==19||i==20)
          axis.col(0) = root0.col(0);
        else
          axis.col(0) = shoul0.col(0);
        axis.col(1) = axis.col(2).cross(axis.col(0)).normalized();
        axis.col(0) = axis.col(1).cross(axis.col(2)).normalized();
      }
      ROT0.push_back(axis);
    }
  }

  //2. get posture data
  std::vector<std::map<int, std::pair<RotationList, Eigen::RowVector3d>>> posture; // all posture data
  std::vector<int> tgf2zed = {0, 1, 2, 4, 5, 6, 7, 3, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24, 9, 16, 21, 25, 27};
  std::map<int, std::pair<RotationList, double>> prevvQ; // posture data for each ID obtained from the latest frames
  std::map<int, std::pair<int, int>> stat; // map to count not detected data
  Eigen::VectorXi P;
  igl::directed_edge_parents(BE, P);
  
  // FUNCTION TO CALCULATE JOINT COORDINATES ACCORDING TO THE VQ
  std::function<Eigen::MatrixXd(const RotationList&)> GetNewC = [&](const RotationList& vQ)->Eigen::MatrixXd{
    Eigen::MatrixXd C1 = C;
    for(int i=1;i<BE.rows();i++)
      C1.row(BE(i, 1)) = (vQ[i]*(C.row(BE(i, 1)) - C.row(BE(i, 0))).transpose()).transpose() + C1.row(BE(i, 0));
    C1.row(BE(0, 1)) = (vQ[0]*(C.row(BE(0, 1)) - C.row(BE(0, 0))).transpose()).transpose() + C1.row(BE(0, 0));
    return C1;
  };

  // FUNCTION TO MODIFY THE VQ (SECONDARY MODIFICATION)
  std::function<void(RotationList&)> FixvQ = [&](RotationList& vQ){
    //ANGLE LIMITATION
    std::vector<int> targets = {7, 13};
    for(int i:targets) //hands < 60 deg
    {
      Eigen::AngleAxisd rQ(vQ[P(i)].inverse()*vQ[i]);
      if(rQ.angle()>0.3*M_PI) vQ[i] = vQ[P(i)]*Eigen::AngleAxisd(0.3*M_PI, rQ.axis());
    }
    //neck
    targets = {17, 21, 0}; 
    for(int i:targets) //feet&neck < 40 deg
    {
      Eigen::AngleAxisd rQ(vQ[P(i)].inverse()*vQ[i]);
      if(rQ.angle()>0.7) vQ[i] = vQ[P(i)]*Eigen::AngleAxisd(0.7, rQ.axis());
    }
    Eigen::MatrixXd C1 = GetNewC(vQ);
    
    //twist fix
    targets = {5, 11, 15, 19};
    Eigen::VectorXi D = Eigen::VectorXi::Constant(P.rows(), -1);
    for(int i=0;i<P.rows();i++)
      if(P(i)>=0) D(P(i)) = i;
    for(int targetBone:targets)
    {
      int end;
      std::vector<int> d;
      for(int i=targetBone;;i=D(i))
      {
        d.push_back(i);
        if(D(i)<0){end = i; break;}
      }
      Eigen::Quaterniond q0 = vQ[P(targetBone)];
      Eigen::MatrixXd C2 = C * q0.matrix().transpose(); 
      Eigen::Vector3d v0 = (C2.row(BE(end, 1))-C2.row(BE(end, 0))).normalized();
      Eigen::Vector3d v1 = (C1.row(BE(end, 1))-C1.row(BE(end, 0))).normalized();
      Eigen::Quaterniond q;
      q = (vQ[end]*q0.inverse() * q.setFromTwoVectors(v0, v1).inverse()).normalized(); //twist
      double itvl = Eigen::AngleAxisd(q).angle() / (double)(d.size());
      if(Eigen::AngleAxisd(q).axis().dot(v1)<0) itvl = -itvl;
      double angle = itvl;
      for(int i:d) 
      {
        v0 = (C2.row(BE(i, 1))-C2.row(BE(i, 0))).normalized();
        v1 = (C1.row(BE(i, 1))-C1.row(BE(i, 0))).normalized();
        vQ[i] = Eigen::AngleAxisd(angle, v1)*q.setFromTwoVectors(v0, v1)*q0;
        angle += itvl;
      }
    }
  };

  // 3. loop to calculate vQ
  Eigen::MatrixXd bbox(2, 3); // matrix to set the view range
  for(size_t f=0;f<frameData.size();f++)
  {
    std::map<int, std::pair<RotationList, Eigen::RowVector3d>> thisFrame; // key: ID, posture data: <vQ, neck position>
    for(auto p:frameData[f])
    {
      stat[p.first].first++;
      //if the skeleton is empty use previous valid frame
      if(p.second.rows()!=35)
      {
        if(prevvQ.find(p.first)==prevvQ.end()) continue; //if there is no valid previous frame data, continue
        thisFrame[p.first] = std::make_pair(prevvQ[p.first].first, Eigen::RowVector3d(p.second(0, 0), p.second(0, 1), prevvQ[p.first].second));
        bbox(0, 0) = bbox(0, 0)>p.second(0, 0)?bbox(0, 0):p.second(0, 0);
        bbox(0, 1) = bbox(0, 1)>p.second(0, 1)?bbox(0, 1):p.second(0, 1);
        bbox(1, 0) = bbox(1, 0)<p.second(0, 0)?bbox(1, 0):p.second(0, 0);
        bbox(1, 1) = bbox(1, 1)<p.second(0, 1)?bbox(1, 1):p.second(0, 1);
        continue;
      }
      stat[p.first].second++;

      //extract required joint position data
      Eigen::MatrixXd C1(23, 3);
      for(int i=0;i<C1.rows();i++)
        C1.row(i) = p.second.row(tgf2zed[i]);

      //calculate bone direction vectors
      Eigen::MatrixXd BV1 = Eigen::MatrixXd::Zero(BE.rows(), 3); 
      for(int i=0;i<BE.rows();i++)
      {
        if(C1(BE(i,0),2)>-1000 && C1(BE(i,1),2)>-1000)
          BV1.row(i) = (C1.row(BE(i, 1)) - C1.row(BE(i, 0))).normalized();
      }

      //calculate exceptional rotations
      Eigen::Matrix3d root1; //root1 should be calculated for all the cases
      if(C1(0, 2)<-1000) // if root is nan
      {
        if(C1(12, 2)>-1000 && C1(15, 2)>-1000) root1.col(0) = (C1.row(12)-C1.row(16)).normalized(); // toLhip
        else root1.col(0) = (C1.row(3)-C1.row(8)).normalized();
        root1.col(2) << 0, 0, 1; // root-spineN
      }
      else
      {
        if(C1(12, 2)>-1000) root1.col(0) = (C1.row(12)-C1.row(0)).normalized(); // toLhip
        else if(C1(15, 2)>-1000) root1.col(0) = (C1.row(0)-C1.row(15)).normalized();
        else root1.col(0) = (C1.row(3)-C1.row(8)).normalized();
        root1.col(2) = BV1.row(1); // root-spineN
      }
      root1.col(1) = root1.col(2).cross(root1.col(0)).normalized();
      root1.col(0) = root1.col(1).cross(root1.col(2)).normalized();
      Eigen::Matrix3d shoul1;
      shoul1.col(0) = (C1.row(3)-C1.row(8)).normalized(); // toLshoul
      shoul1.col(2) = BV1.row(3); // spineC-neck --> always obtainable
      shoul1.col(1) = shoul1.col(2).cross(shoul1.col(0)).normalized();
      shoul1.col(0) = shoul1.col(1).cross(shoul1.col(2)).normalized();
      Eigen::Matrix3d chest1;
      chest1.col(0) = (root1.col(0) + shoul1.col(0)).normalized();
      chest1.col(2) = BV1.row(2);  // always obtainable
      chest1.col(1) = chest1.col(2).cross(chest1.col(0)).normalized();
      chest1.col(0) = chest1.col(1).cross(chest1.col(2)).normalized();
      Eigen::Matrix3d head1;
      bool estimatable = true;
      if(p.second(28, 2)>-1000 && p.second(30, 2)>-1000) head1.col(0) = (p.second.row(28)-p.second.row(30)).normalized(); // eye line
      else if(p.second(29, 2)>-1000 && p.second(31, 2)>-1000) head1.col(0) = (p.second.row(29)-p.second.row(31)).normalized(); 
      else estimatable = false;
      if(estimatable){
        if(C1(22, 2)>-1000) head1.col(2) = (C1.row(22)-C1.row(7)).normalized();
        else head1.col(2) = ((p.second.row(29)+p.second.row(31))*0.5-C1.row(7)-root1.col(1).transpose()*9.).normalized();
        head1.col(1) = head1.col(2).cross(head1.col(0)).normalized();
        head1.col(2) = head1.col(0).cross(head1.col(1)).normalized();
      } else estimatable = false;

      Eigen::Quaterniond rootQ(root1*root0.inverse());
      Eigen::Quaterniond shoulQ(shoul1*shoul0.inverse());
      Eigen::Quaterniond chestQ(chest1*chest0.inverse());

      ////////////////fix BV (primatry modification) --> secondary modification is in the function FixvQ
#ifndef ORIGINAL
      std::vector<int> targets = {15, 16, 19, 20};
      if(prevvQ.find(p.first)!=prevvQ.end()) // missing leg BV
      {
        for(auto i:targets)
        {
          if(BV1.row(i).squaredNorm()<0.1)
            BV1.row(i) = prevvQ[p.first].first[i]*Eigen::Vector3d(BV.row(i));
        }
      }
      //upper leg
      double rightDeg = acos(root1.col(0).dot(BV1.row(19)))-0.5*M_PI;
      if(rightDeg<0) BV1.row(19) = (BV1.row(19)-sin(-rightDeg)*root1.col(0).transpose()).normalized();
      double leftDeg = 0.5*M_PI - acos(root1.col(0).dot(BV1.row(15)));
      if(leftDeg<0) BV1.row(15) = (BV1.row(15)+sin(-leftDeg)*root1.col(0).transpose()).normalized();
      targets = {15, 19};
      for(auto i:targets)
      {
        double deg = acos(BV1.row(i).dot(-Eigen::RowVector3d::UnitZ()));
        if(deg>1./6.*M_PI) BV1.row(i) = Eigen::AngleAxisd(1./6.*M_PI-deg, Eigen::Vector3d(BV1.row(i)).cross(Eigen::Vector3d::UnitZ()))*Eigen::Vector3d(BV1.row(i));
      }

      //lower leg
      Eigen::Vector3d rightN = Eigen::Vector3d(BV1.row(19)).cross(root1.col(2)).normalized();
      Eigen::Vector3d leftN = root1.col(2).cross(Eigen::Vector3d(BV1.row(15))).normalized();
      rightDeg = 0.5*M_PI - acos(rightN.dot(BV1.row(20)));
      leftDeg = 0.5*M_PI - acos(leftN.dot(BV1.row(16)));
      if(f==98) std::cout<<p.first<<" "<<rightDeg/M_PI*180.<<std::endl;
      if(abs(rightDeg)>0.05) BV1.row(20) = (Eigen::AngleAxisd(rightDeg-0.05, rightN.cross(Eigen::Vector3d(BV1.row(20))).normalized())*Eigen::Vector3d(BV1.row(20))).transpose();
      if(abs(leftDeg)>0.05) BV1.row(16) = (Eigen::AngleAxisd(leftDeg-0.05, leftN.cross(Eigen::Vector3d(BV1.row(16))))*Eigen::Vector3d(BV1.row(16))).transpose();
      rightN = Eigen::Vector3d(-BV1.row(19)).cross(rightN).normalized();
      leftN = leftN.cross(Eigen::Vector3d(-BV1.row(15))).normalized();
      rightDeg = rightN.dot(BV1.row(20));
      leftDeg = leftN.dot(BV1.row(16));
      if(rightDeg>0) BV1.row(20) = BV1.row(20)-rightN.transpose()*rightDeg;
      if(leftDeg>0) BV1.row(16) = BV1.row(16)-leftN.transpose()*leftDeg;
#endif
      RotationList vQ;
      for(int i=0;i<BE.rows();i++)
      {
        if(BE(i,0)==0) vQ.push_back(rootQ);
        else if(BE(i,0)==2 || BE(i,0)==3 || BE(i,0)==8) vQ.push_back(shoulQ);
        else if(BE(i,0)==1) vQ.push_back(chestQ);
        else if(BE(i,0)==7)
        {
          if(estimatable) vQ.push_back(Eigen::Quaterniond(head1*head0.inverse()));
          else if(prevvQ.find(p.first)!=prevvQ.end()) vQ.push_back(shoulQ*prevvQ[p.first].first[P(i)].inverse()*prevvQ[p.first].first[i]);
          else vQ.push_back(Eigen::Quaterniond::Identity());
        }
        else if(BV1.row(i).squaredNorm()>0.1)
        {
          Eigen::Matrix3d axis;
          if(i==17||i==21){
            axis.col(1) = -BV1.row(i);
            axis.col(2) = -BV1.row(i-1);
            axis.col(0) = axis.col(1).cross(axis.col(2)).normalized();
            axis.col(2) = axis.col(0).cross(axis.col(1)).normalized();
          }
          else if(i==7||i==13){
            axis.col(2) = -BV1.row(i);
            if(i==7) axis.col(1) = Eigen::Vector3d(p.second.row(9)-p.second.row(10)).normalized();
            else axis.col(1) = Eigen::Vector3d(p.second.row(16)-p.second.row(17)).normalized();
            axis.col(0) = axis.col(1).cross(axis.col(2)).normalized();
            axis.col(1) = axis.col(2).cross(axis.col(0)).normalized();
          }
          else{
            axis.col(2) = -BV1.row(i);
            if(i==15||i==16||i==19||i==20)
              axis.col(0) = root1.col(0);
            else
              axis.col(0) = shoul1.col(0);
            axis.col(1) = axis.col(2).cross(axis.col(0)).normalized();
            axis.col(0) = axis.col(1).cross(axis.col(2)).normalized();
          }
          vQ.push_back(Eigen::Quaterniond(axis*ROT0[i].inverse()));
        }
        else if(prevvQ.find(p.first)!=prevvQ.end()) vQ.push_back(vQ[P(i)]*prevvQ[p.first].first[P(i)].inverse()*prevvQ[p.first].first[i]);
        else vQ.push_back(rootQ);
      }
#ifndef ORIGINAL
      FixvQ(vQ);      
#endif
      thisFrame[p.first] = std::make_pair(vQ, Eigen::RowVector3d(p.second(34, 0), p.second(34, 1), p.second(3, 2)));
      prevvQ[p.first] = std::make_pair(vQ, p.second(3, 2));
      bbox(0, 0) = bbox(0, 0)>p.second(0, 0)?bbox(0, 0):p.second(0, 0);
      bbox(0, 1) = bbox(0, 1)>p.second(0, 1)?bbox(0, 1):p.second(0, 1);
      bbox(1, 0) = bbox(1, 0)<p.second(0, 0)?bbox(1, 0):p.second(0, 0);
      bbox(1, 1) = bbox(1, 1)<p.second(0, 1)?bbox(1, 1):p.second(0, 1);
    }
    posture.push_back(thisFrame);
  }
  std::cout<<"among "<<frameData.size()<<" frame data..."<<std::endl;
  for(auto p:stat)
    std::cout<<p.first<<" : "<<p.second.first<< " pos. / "<<p.second.second<<" skel."<<std::endl;
 
  Eigen::MatrixXd colours(7, 4);
  colours<<1., 0., 0., 1., 1., 127./255., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 75./255., 0., 130./255., 1., 148./255., 0., 211./255., 1.;
  igl::opengl::glfw::Viewer vr;
  Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
  // vr.data().set_edges(C, BE, sea_green);
  vr.data().set_mesh(Vo, Fo);
  vr.data().set_colors(colours.row(0));
  vr.data().point_size = 5;
  vr.data().show_lines = false;
  vr.data().show_overlay_depth = false;
  vr.core().is_animating = false;
  vr.core().animation_max_fps = 5;
  int maxP = stat.rbegin()->first;
  for(int i=1;i<maxP;i++)
  {
    vr.append_mesh();
    vr.data().set_mesh(Vo, Fo);
    vr.data().set_colors(colours.row(i%7));
    vr.data().point_size = 5;
    vr.data().show_lines = false;
    vr.data().show_overlay_depth = false;
  }
  vr.data().show_overlay_depth = false;
  vr.core().camera_eye = Eigen::Vector3f(3., 0, 0);
  vr.core().camera_up = Eigen::Vector3f(0, 0, 1);
  bbox.row(0) += Eigen::RowVector3d(30., 30., 200.);
  bbox.row(1) += Eigen::RowVector3d(-30., -30., -10.);
  vr.core().align_camera_center(bbox);
  int frame(0);//, pid(1);
  vr.callback_pre_draw = [&](igl::opengl::glfw::Viewer _vr)->bool{
    if(!vr.core().is_animating) return false;
    std::cout<<"["<<frame<<"] "<<std::flush;
    if(frame==posture.size()) {frame = 0; vr.core().is_animating = false; return false;}
    for(int p=0;p<maxP;p++)
    {
      if(posture[frame].find(p+1)==posture[frame].end()) {vr.data(p).is_visible = false; continue;}
      vr.data(p).is_visible = true;
      Eigen::MatrixXd C1 = GetNewC(posture[frame][p+1].first);
      // Eigen::RowVector3d trans = C1.row(7);
      C1 = C1.rowwise() +  (posture[frame][p+1].second - C1.row(7));
      std::vector<Eigen::Vector3d>  vT;
      for(int i=0;i<BE.rows();i++)
        vT.push_back(C1.row(BE(i, 0)).transpose()-posture[frame][p+1].first[i]*C.row(BE(i, 0)).transpose());
      Eigen::MatrixXd U;
      myDqs(Vo, weightMap, posture[frame][p+1].first, vT, U);
      // vr.data(p).set_edges(C1, BE, sea_green);
      // vr.data(p).set_points(frameData[frame][p+1], Eigen::RowVector3d(1., 0, 1.));
      vr.data(p).set_vertices(U);
      vr.data(p).compute_normals();
      std::cout<< p+1 <<" "<<std::flush;
    }
    std::cout<<std::endl;
    frame++;
    return false;
  };
  vr.callback_key_down = [&](igl::opengl::glfw::Viewer _vr, unsigned int key, int modifiers)->bool{
  switch (key)
  {
  case ']':
    frame = std::min((int)posture.size()-1, frame+1); 
    break;
  case '[':
    frame = std::max((int)0, frame-1); 
    break;
  case ' ':
    vr.core().is_animating = !vr.core().is_animating;
    return true;
  // case '/':
  //   pid = (++pid%6)+1;
  //   break;
  default:
    return true;
    break;
  }
  std::cout<<"["<<frame<<"] "<<std::flush;
  for(int p=0;p<maxP;p++)
  {
    if(posture[frame].find(p+1)==posture[frame].end()) {vr.data(p).is_visible = false; continue;}
    vr.data(p).is_visible = true;
    Eigen::MatrixXd C1 = GetNewC(posture[frame][p+1].first);
    // Eigen::RowVector3d trans = C1.row(7);
    C1 = C1.rowwise() +  (posture[frame][p+1].second - C1.row(7));
    std::vector<Eigen::Vector3d>  vT;
    for(int i=0;i<BE.rows();i++)
      vT.push_back(C1.row(BE(i, 0)).transpose()-posture[frame][p+1].first[i]*C.row(BE(i, 0)).transpose());
    Eigen::MatrixXd U;
    myDqs(Vo, weightMap, posture[frame][p+1].first, vT, U);
    // vr.data(p).set_edges(C1, BE, sea_green);
    // vr.data(p).set_points(frameData[frame][p+1], Eigen::RowVector3d(1., 0, 1.));
    vr.data(p).set_vertices(U);
    vr.data(p).compute_normals();
    std::cout<< p+1 <<" "<<std::flush;
  }
  std::cout<<std::endl;
  return true;};
  vr.launch();

  return 0;
}
  
  