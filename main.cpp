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
    // Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(b.rows(),G.maxCoeff()+1);
    // for(int i=0;i<BE.rows();i++) bc.col(G(i))+= bc1.col(i);
    // bc1.resize(0, 0);
    // b.conservativeResize(b.rows() + 4);
    // b.bottomRows(4) = Eigen::VectorXi::LinSpaced(4, numVo+12, numVo+15);
    // bc.conservativeResize(b.rows(), bc.cols());
    // bc.bottomRows(4) = Eigen::MatrixXd::Zero(4, bc.cols());
    // bc.block(bc.rows()-4, 3, 4, 1) = Eigen::VectorXd::Ones(4);
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
  std::vector<std::map<int, std::pair<RotationList, Eigen::RowVector3d>>> posture;
  std::vector<int> tgf2zed = {0, 1, 2, 4, 5, 6, 7, 3, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24, 9, 16, 21, 25, 27};
  std::map<int, std::pair<RotationList, double>> prevvQ;
  std::map<int, std::pair<int, int>> stat;
  Eigen::VectorXi P;
  igl::directed_edge_parents(BE, P);
  
  std::function<Eigen::MatrixXd(const RotationList&)> GetNewC = [&](const RotationList& vQ)->Eigen::MatrixXd{
    Eigen::MatrixXd C1 = C;
    for(int i=1;i<BE.rows();i++)
      C1.row(BE(i, 1)) = (vQ[i]*(C.row(BE(i, 1)) - C.row(BE(i, 0))).transpose()).transpose() + C1.row(BE(i, 0));
    C1.row(BE(0, 1)) = (vQ[0]*(C.row(BE(0, 1)) - C.row(BE(0, 0))).transpose()).transpose() + C1.row(BE(0, 0));
    return C1;
  };

  // std::function<void(Eigen::MatrixXd&, Eigen::MatrixXd&)> FixC1 = [&](Eigen::MatrixXd& C2){
  //   //arm/legs fix
  //   Eigen::Vector3d front = Eigen::Vector3d(BV2.row(14)).cross(Eigen::Vector3d(C2.row(1)-C2.row(0))).normalized();
  //   std::vector<int> target = {17, 21};
  //   for(int i:target)
  //   {
  //     double dot = Eigen::Vector3d(BV2.row(i)).dot(front);
  //     if(dot>0.8) continue;
  //     // BV2.row(i) = Eigen::AngleAxisd(acos(0.8-dot), Eigen::Vector3d(BV2.row(i)).cross(front)) * BV2.row(i);
  //     C2.row(BE(i, 1)) = C2.row(BE(i, 0)) + Eigen::AngleAxisd(acos(0.8-dot), Eigen::Vector3d(BV2.row(i)).cross(front)) * (C2.row(BE(i, 1))-C2.row(BE(i, 0)));
  //   }
  // };

  std::function<void(RotationList&)> FixvQ = [&](RotationList& vQ){
    std::vector<int> targets = {7, 13, 17, 21};
    for(int i:targets) //keep hands and feet
    {
      Eigen::AngleAxisd rQ(vQ[P(i)].inverse()*vQ[i]);
      if(rQ.angle()>60./M_PI) vQ[i] = vQ[P(i)]*Eigen::AngleAxisd(60./M_PI, rQ.axis());
    }
    Eigen::MatrixXd C1 = GetNewC(vQ);
    // Eigen::MatrixXd BV1(BE.rows(), 3);
    // for(int i=0;i<BE.rows();i++)
    //   BV1.row(i) = (C1.row(BE(i, 1))-C1.row(BE(i, 0))).normalized();
    // Eigen::Matrix3d rootC = vQ[1].matrix();
    
    // // //legs
    // Eigen::Vector3d legCross = Eigen::Vector3d(BV1.row(19)).cross(Eigen::Vector3d(BV1.row(15)));
    // if(legCross.dot(rootC.col(1))>0) // if legs are crossed
    // {
    //   // if(BV1.row(19).dot(rootC.col(1))>0) vQ[19] = vQ[1];
    //   // if(BV1.row(15).dot(rootC.col(1))<0) vQ[15] = vQ[1];
    //   vQ[19] = vQ[1];
    //   vQ[15] = vQ[1];
    // }
    // C1 = GetNewC(vQ);
    // // Eigen::Vector3d lleg = -(vQ[15]*vQ[1].inverse()).matrix().col(2);
    // Eigen::Vector3d rleg = -(vQ[19]*vQ[1].inverse()).matrix().col(2);

    // targets = {15, 19};
    // Eigen::Matrix3d root1 = vQ[1].matrix();
    // for(int i:targets)
    // {
    //   Eigen::Vector3d bv = (C1.row(BE(i, 1))-C1.row(BE(i, 0))).normalized();
    //   double dotX = bv.dot(root1.col(0));
    //   if((i==15) && (dotX<0)) {vQ[i] = Eigen::AngleAxisd(acos(dotX-0.5*M_PI),-root1.col(1))*vQ[i]; std::cout<<acos(dotX-0.5*M_PI)*180./M_PI<<std::endl;}
    //   else if((i==19) && (dotX>0)) {vQ[i] =  Eigen::AngleAxisd(acos(0.5*M_PI-dotX), root1.col(1))*vQ[i]; std::cout<<acos(0.5*M_PI-dotX)*180./M_PI<<std::endl;}
      
    // }
    // C1 = GetNewC(vQ);
    // Eigen::Vector3d lleg = -(vQ[15]*vQ[1].inverse()).matrix().col(2);
    // Eigen::Vector3d rleg = -(vQ[19]*vQ[1].inverse()).matrix().col(2);
    // if(rleg.cross(lleg).dot(Eigen::Vector3d::UnitZ())>0)
    // {
    //   vQ[15] = vQ[1];
    //   vQ[19] = vQ[1];
    // }
    // knee
    // Eigen::Vector3d normal = (vQ[19]*vQ[1].inverse()).matrix().col(0).normalized();
    // Eigen::Vector3d bv = Eigen::Vector3d(C1.row(17)-C1.row(16)).normalized();
    // double dot = normal.dot(bv);
    // double sin2 = 1-dot*dot;
    // if(sin2>0.25)
    // {
    //   Eigen::Vector3d proj =  bv - normal * dot;
    //   vQ[20] = Eigen::AngleAxisd(1./6.*M_PI-asin(sqrt(sin2)),normal.cross(proj))*vQ[20];
    // }
    // Eigen::Vector3d rknee = -(vQ[20]*vQ[1].inverse()).matrix().col(1);
    // double dot = (rknee.cross(-vQ[1].matrix().col(1))).normalized().dot(-vQ[1].matrix().col(2).normalized());
    // if(dot>0) vQ[20] = Eigen::AngleAxisd(acos(dot),vQ[1].matrix().col(1).normalized())*vQ[20];
    ///////////////////////////////////////////////////////////
    
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
      // for(int i:d) vQ2[i] = vQ2[i]*q0.inverse();
      // vQ2[P(targetBone)] = Eigen::Quaterniond::Identity();
      Eigen::Vector3d v0 = (C2.row(BE(end, 1))-C2.row(BE(end, 0))).normalized();
      Eigen::Vector3d v1 = (C1.row(BE(end, 1))-C1.row(BE(end, 0))).normalized();
      Eigen::Quaterniond q;
      q = (vQ[end]*q0.inverse() * q.setFromTwoVectors(v0, v1).inverse()).normalized(); //twist
      // double totAng = Eigen::AngleAxisd(q).angle();
      // totAng = acos(cos(totAng));
      // if(Eigen::AngleAxisd(q).angle()>M_PI) std::cout<<Eigen::AngleAxisd(q).angle()<<std::endl;
      double itvl = Eigen::AngleAxisd(q).angle() / (double)(d.size());
      if(Eigen::AngleAxisd(q).axis().dot(v1)<0) itvl = -itvl;
      double angle = itvl;
      for(int i:d) 
      {
        v0 = (C2.row(BE(i, 1))-C2.row(BE(i, 0))).normalized();
        v1 = (C1.row(BE(i, 1))-C1.row(BE(i, 0))).normalized();
        vQ[i] = Eigen::AngleAxisd(angle, v1)*q.setFromTwoVectors(v0, v1)*q0;
        // vQ2[i] = Eigen::AngleAxisd(angle, v1)*q.setFromTwoVectors(v0, v1);
        angle += itvl;
        // if(P(i)<0) {vQ[i] = vQ2[i]; continue;}
        // vQ[i] = vQ2[i]*q0;
      }
    }
  };

  Eigen::MatrixXd bbox(2, 3);
  for(size_t f=0;f<frameData.size();f++)
  {
    std::map<int, std::pair<RotationList, Eigen::RowVector3d>> thisFrame;
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

      Eigen::MatrixXd BV1 = Eigen::MatrixXd::Zero(BE.rows(), 3); 
      for(int i=0;i<BE.rows();i++)
      {
        if(C1(BE(i,0),2)>-1000 && C1(BE(i,1),2)>-1000)
          BV1.row(i) = (C1.row(BE(i, 1)) - C1.row(BE(i, 0))).normalized();
      }

      Eigen::Matrix3d root1;
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
      shoul1.col(2) = BV1.row(3); // spineC-neck
      shoul1.col(1) = shoul1.col(2).cross(shoul1.col(0)).normalized();
      shoul1.col(0) = shoul1.col(1).cross(shoul1.col(2)).normalized();
      Eigen::Matrix3d chest1;
      chest1.col(0) = (root1.col(0) + shoul1.col(0)).normalized();
      chest1.col(2) = BV1.row(2); 
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

      ////////////////fix BV
      //upper leg
      double rightDeg = acos(root1.col(0).dot(BV1.row(19)))-0.5*M_PI;
      double leftDeg = 0.5*M_PI - acos(root1.col(0).dot(BV1.row(15)));
      if(rightDeg<0) BV1.row(19) = (BV1.row(19)-sin(-rightDeg)*root1.col(0).transpose()).normalized();
      if(leftDeg<0) BV1.row(15) = (BV1.row(15)+sin(-leftDeg)*root1.col(0).transpose()).normalized();
      //lower leg
      rightDeg = acos(root1.col(0).dot(BV1.row(20)))-0.5*M_PI;
      leftDeg = 0.5*M_PI - acos(root1.col(0).dot(BV1.row(16)));
      if(rightDeg<0) BV1.row(20) = (BV1.row(20)-sin(-rightDeg)*root1.col(0).transpose()).normalized();
      if(leftDeg<0) BV1.row(16) = (BV1.row(16)+sin(-leftDeg)*root1.col(0).transpose()).normalized();
   
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
            axis.col(2) = root1.col(2);
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
      FixvQ(vQ);      
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
  Eigen::MatrixXi BE_zed(10, 2);
  BE_zed<<0, 22, 0, 18, 22, 23, 23, 24, 24, 25, 24, 33, 18, 19, 19, 20, 20, 21, 20, 32;
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
  // vr.data().is_visible = false;
  int maxP = stat.rbegin()->first;
  for(int i=1;i<maxP;i++)
  {
    vr.append_mesh();
    vr.data().set_mesh(Vo, Fo);
    vr.data().set_colors(colours.row(i%7));
    vr.data().point_size = 5;
    vr.data().show_lines = false;
    vr.data().show_overlay_depth = false;
    // vr.data().is_visible = false;
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
      vr.data(p).set_points(frameData[frame][p+1], Eigen::RowVector3d(1., 0, 1.));
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
    vr.data(p).set_points(frameData[frame][p+1], Eigen::RowVector3d(1., 0, 1.));
    vr.data(p).set_edges(frameData[frame][p+1], BE_zed, sea_green);
    vr.data(p).set_vertices(U);
    vr.data(p).compute_normals();
    std::cout<< p+1 <<" "<<std::flush;
  }
  std::cout<<std::endl;
  return true;};
  vr.launch();

  return 0;
  
  //////////////////// start frame data /////////////////////////////////
/*  Eigen::MatrixXd randCol = Eigen::MatrixXd::Random(10, 3).cwiseAbs();
  igl::opengl::glfw::Viewer vr;
  Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
  vr.data().set_edges(C, BE, sea_green);
  
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

  return 0;*/
}
  
  