#include <omp.h>
#include <numeric>
#include <iostream>
#include <string>
#include <vector>
#include "numpy/arrayobject.h"
#include "simdjson.h"

constexpr char kId[] = "id";
constexpr char kImages[] = "images";
constexpr char kHeight[] = "height";
constexpr char kWidth[] = "width";
constexpr char kCats[] = "categories";
constexpr char kAnns[] = "annotations";
constexpr char kSegm[] = "segmentation";
constexpr char kImageId[] = "image_id";
constexpr char kCategoryId[] = "category_id";
constexpr char kArea[] = "area";
constexpr char kIsCrowd[] = "iscrowd";
constexpr char kBbox[] = "bbox";
constexpr char kCounts[] = "counts";
constexpr char kScore[] = "score";
constexpr char kSize[] = "size";
constexpr char kCaption[] = "caption";

struct image_struct {
  int64_t id;
  int h;
  int w;
};

typedef struct {
  unsigned long h;
  unsigned long w;
  unsigned long m;
  unsigned int *cnts;
} RLE;

struct anns_struct {
  int64_t image_id;
  int64_t category_id;
  
  int64_t id;
  float area;
  int iscrowd;
  std::vector<float> bbox;
  float score;
  
  // segmentation
  std::vector<std::vector<double>> segm_list;
  std::vector<int> segm_size;
  std::vector<int> segm_counts_list;
  std::string segm_counts_str;
};

// dict type
struct detection_struct {
  std::vector<int64_t> id;
  std::vector<float> area;
  std::vector<int> iscrowd;
  std::vector<std::vector<float>> bbox;
  std::vector<float> score;
  
  std::vector<std::vector<int>> segm_size;
  std::vector<int> ignore;
  std::vector<std::string> segm_counts;
};

// create index results
std::vector<int64_t> imgids;
std::vector<int64_t> catids;
std::vector<image_struct> imgsgt;
std::vector<std::vector<double>> gtbboxes;
std::vector<std::vector<RLE>> gtsegm;

std::vector<detection_struct> gts;
std::vector<detection_struct> dts;

// internal computeiou results
std::vector<std::vector<double>> ious_map;

// global variables within each process
int num_procs = 0;
int proc_id = 0;

double *precision;
double *recall;
double *scores;
size_t len_precision = 0;
size_t len_recall = 0;
size_t len_scores = 0;

template <typename T, typename Comparator = std::greater<T> >
std::vector<size_t> sort_indices(const std::vector<T>& v, Comparator comparator = Comparator()) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
                  [&](size_t i1, size_t i2) { return comparator(v[i1], v[i2]); });
  return indices;
}

template <typename T, typename Comparator = std::greater<T> >
std::vector<size_t> stable_sort_indices(const std::vector<T>& v, Comparator comparator = Comparator()) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(),
                  [&](size_t i1, size_t i2) { return comparator(v[i1], v[i2]); });
  return indices;
}

void accumulate(int T, int A, const int* maxDets, const double* recThrs,
                double* precision, double* recall, double* scores,
                const int K, const int I, const int R, const int M, const int k, const int a,
                const std::vector<std::vector<int64_t>>& gtignore,
                const std::vector<std::vector<double>>& dtignore,
                const std::vector<std::vector<double>>& dtmatches,
                const std::vector<std::vector<double>>& dtscores);

void compute_iou(std::string iouType, int maxDet, int useCats, int nthreads);

static PyObject* cpp_evaluate(PyObject* self, PyObject* args) {

  // there must be at least 1 process to run this program
  if (num_procs<1){
    std::cout << "[cpp_evaluate] Error: num_procs must be >=1" << std::endl;
    return NULL;
  }

  // read arguments
  int useCats;
  PyArrayObject *pareaRngs, *piouThrs_ptr, *pmaxDets, *precThrs; 
  const char* iouType_chars; 

  // evaluate() will use however many proesses create_index uses, 
  // and with the same (pid : data chuck id) mapping
  int nthreads;
  if (!PyArg_ParseTuple(args, "iO!O!O!O!si|", &useCats, 
                          &PyArray_Type, &pareaRngs, &PyArray_Type, &piouThrs_ptr, 
                          &PyArray_Type, &pmaxDets, &PyArray_Type, &precThrs, 
                          &iouType_chars, &nthreads)) {
    std::cout << "[cpp_evaluate] Error: can't parse arguments (must be int, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,str, int)" << std::endl;
    return NULL;
  }
  std::string iouType(iouType_chars);
  if (useCats<=0){
    std::cout << "[cpp_evaluate] Error: useCats must be >0" << std::endl;
    return NULL;
  }
  
  int areaRngs_dim1 = pareaRngs->dimensions[0];
  int areaRngs_dim2 = pareaRngs->dimensions[1];
  double **areaRngs = (double **)malloc((size_t) (areaRngs_dim1*sizeof(double)));
  double *areaRngs_data = (double *)pareaRngs->data;
  for (int i=0; i<areaRngs_dim1; i++)
    areaRngs[i] = areaRngs_data + i * areaRngs_dim2;
  double *iouThrs_ptr = (double *)piouThrs_ptr->data;
  int *maxDets = (int *)pmaxDets->data;
  double *recThrs = (double *)precThrs->data;

  int T = piouThrs_ptr->dimensions[0];
  int A = pareaRngs->dimensions[0];
  int K = catids.size();
  int R = precThrs->dimensions[0];
  int M = pmaxDets->dimensions[0];
  
  const npy_intp len1[] = {T*R*K*A*M};
  const npy_intp len2[] = {T*K*A*M};

  // if precision/recall/scores have been allocated, do not allocate again
  // if they haven't, allocate on the heap and ensure returned PyObject retains value
  if (len_precision != len1[0]) {
    precision = (double *)malloc((size_t)(len1[0]*sizeof(double)));
    len_precision = len1[0];
  }
  if (len_recall != len2[0]) {
    recall = (double *)malloc((size_t)(len2[0]*sizeof(double)));
    len_recall = len2[0];
  }
  if (len_scores != len1[0]) {
    scores = (double *)malloc((size_t)(len1[0]*sizeof(double)));
    len_scores = len1[0];
  }

  // initialize in first run or zero out in subsequent runs
  for (npy_intp i=0; i<len1[0]; i++) {
    precision[i]=0.0D;
    scores[i]=0.0D;
  }
  for (npy_intp i=0; i<len2[0]; i++) {
    recall[i]=0.0D;
  }
  
  // wrap data with PyObject wrapper
  PyObject* pprecision = PyArray_SimpleNewFromData(1, len1, NPY_DOUBLE, precision); 
  PyObject* precall = PyArray_SimpleNewFromData(1, len2, NPY_DOUBLE, recall); 
  PyObject* pscores = PyArray_SimpleNewFromData(1, len1, NPY_DOUBLE, scores); 

  // if gts or dts doesn't exist, no need to evaluate, return zeros as precision/recall/scores
  if (gts.size() > 0 && dts.size() > 0) 
  {

    // compute ious
    int maxDet = maxDets[M-1];
    compute_iou(iouType, maxDet, useCats, nthreads);
    
    // main loop
    #pragma omp parallel for num_threads(nthreads) collapse(2) schedule(guided, 8)
    for(npy_intp a = 0; a < pareaRngs->dimensions[0]; a++) {
      for(size_t c = 0; c < catids.size(); c++) {
        const double aRng0 = areaRngs[a][0];
        const double aRng1 = areaRngs[a][1];
        
        std::vector<std::vector<int64_t>> gtIgnore_list;
        std::vector<std::vector<double>> dtIgnore_list;
        std::vector<std::vector<double>> dtMatches_list;
        std::vector<std::vector<double>> dtScores_list;
        
        for(size_t i = 0; i < imgids.size(); i++) {

          auto& gtsm = gts[c*imgids.size() + i];
          auto& dtsm = dts[c*imgids.size() + i];
          if((gtsm.id.size()==0) && (dtsm.id.size()==0)) 
            continue;
        
          // sizes
          const int T = piouThrs_ptr->dimensions[0];
          const int G = gtsm.id.size();
          const int Do = dtsm.id.size();
          const int D = std::min(Do, maxDet);
          const int I = (G==0||D==0) ? 0 : D;
          
          // arrays
          std::vector<double> gtm(T*G, 0.0);
          gtIgnore_list.push_back(std::vector<int64_t>(G));
          dtIgnore_list.push_back(std::vector<double>(T*D, 0.0));
          dtMatches_list.push_back(std::vector<double>(T*D, 0.0));
          dtScores_list.push_back(std::vector<double>(D));
          
          // pointers
          auto& gtIg = gtIgnore_list.back();
          auto& dtIg = dtIgnore_list.back();
          auto& dtm = dtMatches_list.back();
          auto& dtScores = dtScores_list.back();
          auto ious = (ious_map[c*imgids.size() + i].size() == 0) ? nullptr : ious_map[c*imgids.size() + i].data();
          
          // set ignores
          for (int g = 0; g < G; g++) {
            gtIg[g] = (gtsm.ignore[g] || gtsm.area[g]<aRng0 || gtsm.area[g]>aRng1) ? 1 : 0;
          }
          // get sorting indices
          auto gtind = sort_indices(gtIg, std::less<double>());
          auto dtind = sort_indices(dtsm.score);
          
          if(I != 0) {
            for (int t = 0; t < T; t++) {
              double thresh = iouThrs_ptr[t];
              for (int d = 0; d < D; d++) {
                double iou = thresh < (1-1e-10) ? thresh : (1-1e-10);
                int m = -1;
                for (int g = 0; g < G; g++) {
                  // if this gt already matched, and not a crowd, continue
                  if((gtm[t * G + g]>0) && (gtsm.iscrowd[gtind[g]]==0))
                    continue;
                  // if dt matched to reg gt, and on ignore gt, stop
                  if((m>-1) && (gtIg[gtind[m]]==0) && (gtIg[gtind[g]]==1))
                    break;
                  // continue to next gt unless better match made
                  double val = ious[d + I * gtind[g]];
                  if(val < iou)
                    continue;
                  // if match successful and best so far, store appropriately
                  iou=val;
                  m=g;
                }
                // if match made store id of match for both dt and gt
                if(m ==-1)
                  continue;
                dtIg[t * D + d] = gtIg[gtind[m]];
                dtm[t * D + d]  = gtsm.id[gtind[m]];
                gtm[t * G + m]  = dtsm.id[dtind[d]];
              }
            }
          }
          // set unmatched detections outside of area range to ignore
          for (int d = 0; d < D; d++) {
            float val = dtsm.area[dtind[d]];
            double x3 = (val<aRng0 || val>aRng1);
            for (int t = 0; t < T; t++) {
              double x1 = dtIg[t * D + d];
              double x2 = dtm[t * D + d];
              double res = x1 || ((x2==0) && x3);
              dtIg[t * D + d] = res;
            }
            // store results for given image and category
            dtScores[d] = dtsm.score[dtind[d]];
          }
        }
        // accumulate
        accumulate((int)(piouThrs_ptr->dimensions[0]), (int)(pareaRngs->dimensions[0]), maxDets, recThrs,
                        precision, recall, scores,
                        catids.size(), imgids.size(), 
                        (int)(precThrs->dimensions[0]), (int)(pmaxDets->dimensions[0]), c, a,
                        gtIgnore_list, dtIgnore_list, dtMatches_list, dtScores_list);
      }
    }

    // clear arrays
    ious_map.clear();
    //gts.clear();
    dts.clear();
    free(areaRngs);

  }

  PyObject* l = Py_BuildValue("[iiiii]", T, R, K, A, M); 

  npy_intp dims1[] = {K,A,M,T,R};
  npy_intp dims2[] = {K,A,M,T};
  PyArray_Dims pdims1 = {dims1, 5};
  PyArray_Dims pdims2 = {dims2, 4};
  PyArray_Resize((PyArrayObject*)pprecision, &pdims1, 0, NPY_CORDER); 
  PyArray_Resize((PyArrayObject*)precall, &pdims2, 0, NPY_CORDER);
  PyArray_Resize((PyArrayObject*)pscores, &pdims1, 0, NPY_CORDER);
  
  const npy_intp dims3[] = {imgids.size()};
  const npy_intp dims4[] = {catids.size()};
  PyObject* imgidsret = PyArray_SimpleNewFromData(1, dims3, NPY_INT64, imgids.data());  
  PyObject* catidsret = PyArray_SimpleNewFromData(1, dims4, NPY_INT64, catids.data());  
  
  PyObject* pReturn = Py_BuildValue("(O,O,{s:O,s:O,s:O,s:O})",
                  imgidsret,catidsret,
                  "counts",l,"precision",pprecision,"recall",precall,"scores",pscores);
  
  return pReturn;
}


template <typename T>
std::vector<T> assemble_array(const std::vector<std::vector<T>>& list, size_t nrows, size_t maxDet, const std::vector<size_t>& indices) {
  std::vector<T> q;
  // Need to get N_rows from an entry in order to compute output size
  // copy first maxDet entries from each entry -> array
  for (size_t e = 0; e < list.size(); ++e) {
    auto arr = list[e];
    size_t cols = arr.size() / nrows;
    size_t ncols = std::min(maxDet, cols);
    for (size_t j = 0; j < ncols; ++j) {
      for (size_t i = 0; i < nrows; ++i) {
        q.push_back(arr[i * cols + j]);
      }
    }
  }
  // now we've done that, copy the relevant entries based on indices
  std::vector<T> res(indices.size() * nrows);
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < indices.size(); ++j) {
      res[i * indices.size() + j] = q[indices[j] * nrows + i];
    }
  }

  return res;
}

void accumulate(int T, int A, const int* maxDets, const double* recThrs,
                double* precision, double* recall, double* scores,
                const int K, const int I, const int R, const int M, const int k, const int a,
                const std::vector<std::vector<int64_t>>& gtignore,
                const std::vector<std::vector<double>>& dtignore,
                const std::vector<std::vector<double>>& dtmatches,
                const std::vector<std::vector<double>>& dtscores) 
{
  if (dtscores.size() == 0) return;
  
  int npig = 0;
  for (auto& v: gtignore) {
    npig += count(v.begin(), v.end(), 0);
  }
  if (npig == 0) return;
  
  double eps = 2.220446049250313e-16; //numeric_limits<double>::epsilon();
  for (int m = 0; m < M; ++m) {
    // Concatenate first maxDet scores in each evalImg entry, -ve and sort w/indices
    std::vector<double> dtScores;
    for (size_t e = 0; e < dtscores.size(); ++e) {
      auto score = dtscores[e];
      for (size_t j = 0; j < std::min(score.size(), static_cast<size_t>(maxDets[m])); ++j) {
        dtScores.push_back(score[j]);
      }
    }
  
    // get sorted indices of scores
    auto indices = stable_sort_indices(dtScores);
    std::vector<double> dtScoresSorted(dtScores.size());
    for (size_t j = 0; j < indices.size(); ++j) {
      dtScoresSorted[j] = dtScores[indices[j]];
    }
    
    auto dtm = assemble_array<double>(dtmatches, T, maxDets[m], indices);
    auto dtIg = assemble_array<double>(dtignore, T, maxDets[m], indices);
    
    size_t nrows = indices.size() ? dtm.size()/indices.size() : 0;
    size_t nd = indices.size();
    std::vector<double> tp_sum(nd * nrows);
    std::vector<double> fp_sum(nd * nrows);
    std::vector<double> rc(nd);
    std::vector<double> pr(nd);
    for (size_t t = 0; t < nrows; ++t) {
      size_t tsum = 0, fsum = 0;
      for (size_t j = 0; j < nd; ++j) {
        size_t index = t * nd + j;
        tsum += (dtm[index]) && (!dtIg[index]);
        fsum += (!dtm[index]) && (!dtIg[index]);
        tp_sum[index] = tsum;
        fp_sum[index] = fsum;
      }

      for (size_t j = 0; j < nd; ++j) {
        size_t index = t * nd + j;
        rc[j] = tp_sum[index] / npig;
        pr[j] = tp_sum[index] / (fp_sum[index]+tp_sum[index]+eps);
      }
      
      recall[k*A*M*T + a*M*T + m*T + t] = nd ? rc[nd-1] : 0;
  
      for (size_t i = nd-1; i > 0; --i) {
        if (pr[i] > pr[i-1]) {
          pr[i-1] = pr[i];
        }
      }
  
      size_t inds;
      for (int i = 0; i < R; i++) {
        auto it = lower_bound(rc.begin(), rc.end(), recThrs[i]);
        inds = it - rc.begin();
        size_t index = k*A*M*T*R + a*M*T*R + m*T*R + t*R + i;
        if (inds < nd) {
          precision[index] = pr[inds];
          scores[index] = dtScoresSorted[inds];
        }
      } // i loop
    } // t loop
  } // m loop
}

void bbIou(const double *dt, const double *gt, const int m, const int n, const int *iscrowd, double *o) 
{
  for(int g=0; g<n; g++ ) {
    const double* G = gt+g*4;
    const double ga = G[2]*G[3];
    const double gw = G[2]+G[0];
    const double gh = G[3]+G[1];
    const double crowd = iscrowd!=NULL && iscrowd[g];

    for(int d=0; d<m; d++ ) {
      const double* D = dt+d*4;
      const double da = D[2]*D[3];
      double w = fmin(D[2]+D[0],gw)-fmax(D[0],G[0]);
      double h = fmin(D[3]+D[1],gh)-fmax(D[1],G[1]);
      h = h<=0 ? 0 : h;
      w = w<=0 ? 0 : w;
      double i = w*h;
      double u = crowd ? da : da+ga-i;
      o[g*m+d] = i/u;
    }
  }
}

void rleInit( RLE *R, unsigned long h, unsigned long w, unsigned long m, unsigned int *cnts ) {
  R->h=h;
  R->w=w;
  R->m=m;
  R->cnts = (m==0) ? 0: (unsigned int*)malloc(sizeof(unsigned int)*m);
  if(cnts) {
    for(unsigned long j=0; j<m; j++){
      R->cnts[j]=cnts[j];
    }
  }
}

void rleFree( RLE *R ) {
  free(R->cnts);
  R->cnts=0;
}

void rleFrString(RLE *R, char *s, unsigned long h, unsigned long w ) {
  unsigned long m=0, p=0, k;
  long x;
  int more;
  unsigned int *cnts;
  while( s[m] ){
    m++;
  }
  cnts = (unsigned int*)malloc(sizeof(unsigned int)*m);
  m = 0;
  while( s[p] ) {
    x=0; k=0; more=1;
    while( more ) {
      char c=s[p]-48; x |= (c & 0x1f) << 5*k;
      more = c & 0x20; p++; k++;
      if(!more && (c & 0x10)) x |= -1 << 5*k;
    }
    if(m>2) {
      x += static_cast<long>(cnts[m-2]);
    }
    cnts[m++] = static_cast<unsigned int>(x);
  }
  rleInit(R, h, w, m, cnts);
  free(cnts);
}

unsigned int umin( unsigned int a, unsigned int b ) { return (a<b) ? a : b; }
unsigned int umax( unsigned int a, unsigned int b ) { return (a>b) ? a : b; }

void rleArea( const RLE *R, unsigned long n, unsigned int *a ) {
  for(unsigned long i=0; i<n; i++ ) {
    a[i]=0;
    for(unsigned long j=1; j<R[i].m; j+=2 ) {
      a[i]+=R[i].cnts[j];
    }
  }
}

void rleToBbox( const RLE *R, double* bb, unsigned long n ) {
  for(unsigned long i=0; i<n; i++ ) {
    unsigned int h, w, xp=0, cc=0, t;
    double x, y, xs, ys, xe=0.0D, ye=0.0D; 
    unsigned long j, m;
    h = static_cast<unsigned int>(R[i].h);
    w = static_cast<unsigned int>(R[i].w);
    m = (static_cast<unsigned long>(R[i].m/2))*2;
    xs=w;
    ys=h;
    if(m==0) {
      bb[4*i+0]=bb[4*i+1]=bb[4*i+2]=bb[4*i+3]=0;
      continue;
    }
    for( j=0; j<m; j++ ) {
      cc += R[i].cnts[j];
      t = cc-j%2;
      y = t%h;
      x = (t-y)/h;
      if(j%2==0) {
        xp = x;
      } else if(xp<x) {
        ys = 0;
        ye = h-1;
      }
      xs = umin(xs, x);
      xe = umax(xe, x);
      ys = umin(ys, y);
      ye = umax(ye, y);
    }
    bb[4*i+0] = xs;
    bb[4*i+1] = ys;
    bb[4*i+2] = xe-xs+1;
    bb[4*i+3] = ye-ys+1;
  }
}
void rleIou(const RLE *dt, const RLE *gt, const int m, const int n, const int *iscrowd, double *o ) 
{
  double *db=(double*)malloc(sizeof(double)*m*4);
  double *gb=(double*)malloc(sizeof(double)*n*4);
  rleToBbox(dt, db, m);
  rleToBbox(gt, gb, n);
  bbIou(db, gb, m, n, iscrowd, o);
  free(db);
  free(gb);
  for(int g=0; g<n; g++ ) {
    for(int d=0; d<m; d++ ) {
      if(o[g*m+d]>0) {
        int crowd = iscrowd!=NULL && iscrowd[g];
        if(dt[d].h!=gt[g].h || dt[d].w!=gt[g].w) {
          o[g*m+d]=-1;
          continue;
        }
        unsigned long ka, kb, a, b; uint c, ca, cb, ct, i, u; int va, vb;
        ca=dt[d].cnts[0]; ka=dt[d].m; va=vb=0;
        cb=gt[g].cnts[0]; kb=gt[g].m; a=b=1; i=u=0; ct=1;
        while(ct > 0) {
          c=umin(ca,cb);
          if(va||vb) {
            u+=c;
            if(va&&vb) {
              i+=c;
            }
          }
          ct = 0;
          ca -=c;
          if(!ca && a<ka) {
            ca=dt[d].cnts[a++]; va=!va;
          }
          ct+=ca;
          cb -=c;
          if(!cb && b<kb) {
            cb=gt[g].cnts[b++];
            vb=!vb;
          }
          ct += cb;
        }
        if(i==0) {
          u=1;
        } else if(crowd) {
          rleArea(dt+d, 1, &u);
        }
        o[g*m+d] = static_cast<double>(i)/static_cast<double>(u);
      }
    }
  }
}

void compute_iou(std::string iouType, int maxDet, int useCats, int nthreads) {
  assert(iouType=="bbox"||iouType=="segm");
  assert(useCats > 0);
  
  if (ious_map.size()>0){
    if (proc_id == 0) std::cout << "[compute_iou] IoUs already exist. Clearing vectors..." << std::endl;
    ious_map.clear();
  }
  ious_map.resize(imgids.size() * catids.size());
  
  // compute iou
  #pragma omp parallel for num_threads(nthreads) schedule(guided, 8) collapse(2)
  for(size_t c = 0; c < catids.size(); c++) {
    for(size_t i = 0; i < imgids.size(); i++) {
      const auto gtsm = gts[c*imgids.size() + i];
      const auto dtsm = dts[c*imgids.size() + i];
      const auto G = gtsm.id.size();
      const auto D = dtsm.id.size();
      const int m = std::min(D, static_cast<size_t>(maxDet));
      const int n = G;
      
      if(m==0 || n==0) {
        continue;
      }
      ious_map[c*imgids.size() + i] = std::vector<double>(m*n);
      
      auto inds = sort_indices(dtsm.score);

      if (iouType == "bbox") {
        std::vector<double> d;
        for (auto i = 0; i < m; i++) {
          auto arr = dtsm.bbox[inds[i]];
          for (size_t j = 0; j < arr.size(); j++) {
            d.push_back(static_cast<double>(arr[j]));
          }
        }
  
        bbIou(d.data(), gtbboxes[c*imgids.size()+i].data(), m, n, gtsm.iscrowd.data(), ious_map[c*imgids.size() + i].data());
  
      } else {
        std::vector<RLE> d(m);
        for (auto i = 0; i < m; i++) {
          auto size = dtsm.segm_size[i];
          auto str = dtsm.segm_counts[inds[i]];
          char *val = const_cast<char*>(str.c_str());
          rleFrString(&d[i],val,size[0],size[1]);
          val = NULL;
        }
        
        rleIou(d.data(), gtsegm[c*imgids.size()+i].data(), m, n, gtsm.iscrowd.data(), ious_map[c*imgids.size() + i].data());
        
        for (size_t i = 0; i < d.size(); i++) {free(d[i].cnts);}
      }
    }
  }
}

std::string rleToString( const RLE *R ) {
  /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
  unsigned long i, m=R->m, p=0; long x; int more;
  char *s=(char*)malloc(sizeof(char)*m*6);
  for( i=0; i<m; i++ ) {
    x=(long) R->cnts[i]; if(i>2) x-=(long) R->cnts[i-2]; more=1;
    while( more ) {
      char c=x & 0x1f; x >>= 5; more=(c & 0x10) ? x!=-1 : x!=0;
      if (more) 
        c |= 0x20;
      c+=48;
      s[p++]=c;
    }
  }
  s[p]=0;
  std::string str = std::string(s);
  free(s);
  return str;
}

std::string frUncompressedRLE(std::vector<int> cnts, std::vector<int> size, int h, int w) {
  unsigned int *data = (unsigned int*) malloc(cnts.size() * sizeof(unsigned int));
  for(size_t i = 0; i < cnts.size(); i++) {
    data[i] = static_cast<unsigned int>(cnts[i]);
  }
  RLE R;// = RLE(size[0],size[1],cnts.size(),data);
  R.h = size[0];
  R.w = size[1];
  R.m = cnts.size();
  R.cnts = data;
  std::string str = rleToString(&R);
  free(data);
  return str;
}

int uintCompare(const void *a, const void *b) {
  unsigned int c=*((unsigned int*)a), d=*((unsigned int*)b); return c>d?1:c<d?-1:0;
}

void rleFrPoly(RLE *R, const double *xy, int k, int h, int w ) {
  /* upsample and get discrete points densely along entire boundary */
  int j, m=0;
  double scale=5;
  unsigned int *a, *b;
  int *x = (int*)malloc(sizeof(int)*(k+1));
  int *y = (int*)malloc(sizeof(int)*(k+1));
  for(j=0; j<k; j++) 
    x[j] = static_cast<int>(scale*xy[j*2+0]+.5);
  x[k] = x[0];
  for(j=0; j<k; j++) 
    y[j] = static_cast<int>(scale*xy[j*2+1]+.5);
  y[k] = y[0];
  for(j=0; j<k; j++) 
    m += umax(abs(x[j]-x[j+1]),abs(y[j]-y[j+1]))+1;
  int *u=(int*)malloc(sizeof(int)*m);
  int *v=(int*)malloc(sizeof(int)*m);
  m = 0;
  for( j=0; j<k; j++ ) {
    int xs=x[j], xe=x[j+1], ys=y[j], ye=y[j+1], dx, dy, t, d;
    int flip; double s; dx=abs(xe-xs); dy=abs(ys-ye);
    flip = (dx>=dy && xs>xe) || (dx<dy && ys>ye);
    if(flip) { t=xs; xs=xe; xe=t; t=ys; ys=ye; ye=t; }
    s = dx>=dy ? static_cast<double>(ye-ys)/dx : static_cast<double>(xe-xs)/dy;
    if(dx>=dy) for( d=0; d<=dx; d++ ) {
      t=flip?dx-d:d; u[m]=t+xs; v[m]=static_cast<int>(ys+s*t+.5); m++;
    } else for( d=0; d<=dy; d++ ) {
      t=flip?dy-d:d; v[m]=t+ys; u[m]=static_cast<int>(xs+s*t+.5); m++;
    }
  }
  /* get points along y-boundary and downsample */
  free(x); free(y); k=m; m=0; double xd, yd;
  x=(int*)malloc(sizeof(int)*k); y=(int*)malloc(sizeof(int)*k);
  for( j=1; j<k; j++ ) if(u[j]!=u[j-1]) {
    xd=static_cast<double>(u[j]<u[j-1]?u[j]:u[j]-1); xd=(xd+.5)/scale-.5;
    if( floor(xd)!=xd || xd<0 || xd>w-1 ) continue;
    yd=static_cast<double>(v[j]<v[j-1]?v[j]:v[j-1]); yd=(yd+.5)/scale-.5;
    if(yd<0) yd=0; else if(yd>h) yd=h; yd=ceil(yd);
    x[m]=static_cast<int>(xd); y[m]=static_cast<int>(yd); m++;
  }
  /* compute rle encoding given y-boundary points */
  k=m; a=(unsigned int*)malloc(sizeof(unsigned int)*(k+1));
  for( j=0; j<k; j++ ) a[j]=static_cast<unsigned int>(x[j]*static_cast<int>(h)+y[j]);
  a[k++]=static_cast<unsigned int>(h*w); free(u); free(v); free(x); free(y);
  qsort(a,k,sizeof(unsigned int),uintCompare); unsigned int p=0;
  for( j=0; j<k; j++ ) { unsigned int t=a[j]; a[j]-=p; p=t; }
  b=(unsigned int*)malloc(sizeof(unsigned int)*k); j=m=0; b[m++]=a[j++];
  while(j<k) if(a[j]>0) b[m++]=a[j++]; else {
    j++; if(j<k) b[m-1]+=a[j++]; }
  rleInit(R,h,w,m,b); free(a); free(b);
}

void rleMerge( const RLE *R, RLE *M, unsigned long n, int intersect ) {
  unsigned int *cnts, c, ca, cb, cc, ct; int v, va, vb, vp;
  unsigned long i, a, b, h=R[0].h, w=R[0].w, m=R[0].m; RLE A, B;
  if(n==0) { rleInit(M,0,0,0,0); return; }
  if(n==1) { rleInit(M,h,w,m,R[0].cnts); return; }
  cnts = (unsigned int*)malloc(sizeof(unsigned int)*(h*w+1));
  for( a=0; a<m; a++ ) cnts[a]=R[0].cnts[a];
  for( i=1; i<n; i++ ) {
    B=R[i]; if(B.h!=h||B.w!=w) { h=w=m=0; break; }
    rleInit(&A,h,w,m,cnts); ca=A.cnts[0]; cb=B.cnts[0];
    v=va=vb=0; m=0; a=b=1; cc=0; ct=1;
    while( ct>0 ) {
      c=umin(ca,cb); cc+=c; ct=0;
      ca-=c; if(!ca && a<A.m) { ca=A.cnts[a++]; va=!va; } ct+=ca;
      cb-=c; if(!cb && b<B.m) { cb=B.cnts[b++]; vb=!vb; } ct+=cb;
      vp=v; if(intersect) v=va&&vb; else v=va||vb;
      if( v!=vp||ct==0 ) { cnts[m++]=cc; cc=0; }
    }
    rleFree(&A);
  }
  rleInit(M,h,w,m,cnts); free(cnts);
}

void rlesInit( RLE **R, unsigned long n ) {
  unsigned long i; *R = (RLE*) malloc(sizeof(RLE)*n);
  for(i=0; i<n; i++) rleInit((*R)+i,0,0,0,0);
}

std::string frPoly(std::vector<std::vector<double>> poly, int h, int w) {
  size_t n = poly.size();
  RLE *Rs;
  rlesInit(&Rs,n);
  for (size_t i = 0; i < n; i++) {
    double* p = (double*)malloc(sizeof(double)*poly[i].size());
    for (size_t j = 0; j < poly[i].size(); j++) {
      p[j] = static_cast<double>(poly[i][j]);
    }
    rleFrPoly(&Rs[i],p,int(poly[i].size()/2),h,w);
    free(p);
  }

  RLE R;
  int intersect = 0;
  rleMerge(Rs, &R, n, intersect);
  std::string str = rleToString(&R);
  for (size_t i = 0; i < n; i++) {free(Rs[i].cnts);}
  free(Rs);
  return str;
}

unsigned int area(std::vector<int>& size, std::string& counts) {
  // _frString
  RLE *Rs;
  rlesInit(&Rs,1);
  char *str = const_cast<char*>(counts.c_str());
  rleFrString(&Rs[0],str,size[0],size[1]);
  str = NULL;
  unsigned int a;
  rleArea(Rs, 1, &a);
  for (size_t i = 0; i < 1; i++) {free(Rs[i].cnts);}
  free(Rs);
  return a;
}

std::vector<float> toBbox(std::vector<int>& size, std::string& counts) {
  // _frString
  RLE *Rs;
  rlesInit(&Rs,1);
  char *str = const_cast<char*>(counts.c_str());
  rleFrString(&Rs[0],str,size[0],size[1]);
  str = NULL;

  std::vector<double> bb(4*1);
  rleToBbox(Rs, bb.data(), 1);
  std::vector<float> bbf(bb.size());
  for (size_t i = 0; i < bb.size(); i++) {
    bbf[i] = static_cast<float>(bb[i]);
  }
  for (size_t i = 0; i < 1; i++) {free(Rs[i].cnts);}
  free(Rs);
  return bbf;
}

void annToRLE(anns_struct& ann, std::vector<std::vector<int>> &size, std::vector<std::string> &counts, int h, int w) {
  auto is_segm_list = ann.segm_list.size()>0;
  auto is_cnts_list = is_segm_list ? 0 : ann.segm_counts_list.size()>0;

  if (is_segm_list) {
    std::vector<int> segm_size{h,w};
    auto cnts = ann.segm_list;
    auto segm_counts = frPoly(cnts, h, w);
    size.push_back(segm_size);
    counts.push_back(segm_counts);
  } else if (is_cnts_list) {
    auto segm_size = ann.segm_size;
    auto cnts = ann.segm_counts_list;
    auto segm_counts = frUncompressedRLE(cnts, segm_size, h, w);
    size.push_back(segm_size);
    counts.push_back(segm_counts);
  } else {
    auto segm_size = ann.segm_size;
    auto segm_counts = ann.segm_counts_str;
    size.push_back(segm_size);
    counts.push_back(segm_counts);
  }
}

static PyObject* cpp_create_index(PyObject* self, PyObject* args) 
{
  /* this function takes a JSON file and reads it into gts
   * the JSON file should have keys, 'images', 'categories' and 'annotations'
   * the 'images' field should have subfields, 'id', 'height', and 'width'
   * the 'categories' field should have subfield, 'id'
   * the 'annotations' field should have the following structure: 
   * [
   *    {'image_id': int/int64, 'category_id': int/int64, 
   *    'segmentation': xxx,
   *    'bbox': [float, float, float, float], 'score': float/double},
   *    { ... }
   *  ]
   * the 'segmentation' filed could take one of the following forms: 
   *    'segmentation': [[int, ...], ...]
   *    'segmentation': {'size': [int, int], 'counts': [int, ...]} 
   *    'segmentation': {'size': [int, int], 'counts': std::string} 
   * */ 

  const char *annotation_file;
  int nthreads;
  if (!PyArg_ParseTuple(args, "siii|", &annotation_file, &num_procs, &proc_id, &nthreads)) {
    std::cout << "[cpp_create_index] Error: can't parse the argument (must be std::string)" << std::endl;
    return NULL;
  }
  // there must be at least 1 process to run this program
  if (num_procs<1){
    std::cout << "[cpp_create_index] Error: num_procs must be >=1" << std::endl;
    return NULL;
  }
  
  if (imgsgt.size()>0 || imgids.size()>0 || catids.size()>0) {
    if (proc_id == 0) std::cout << "[cpp_create_index] Ground truth annotations already exist. Clearing vectors..." << std::endl;
    imgsgt.clear();
    imgids.clear();
    catids.clear();
    gtbboxes.clear();
    gtsegm.clear();
    gts.clear();
  }

  simdjson::dom::parser parser;
  simdjson::dom::element dataset = parser.load(annotation_file);

  simdjson::dom::array imgs = dataset[kImages];
  for (simdjson::dom::object image: imgs) {
    image_struct img;
    img.id = image[kId]; 
    img.h = static_cast<int>(image[kHeight].get_int64()); 
    img.w = static_cast<int>(image[kWidth].get_int64()); 
    imgsgt.push_back(img);
    imgids.push_back(img.id);
  }
  
  simdjson::dom::array cats = dataset[kCats];
  std::vector<int64_t> catids_tmp;
  for (simdjson::dom::object cat: cats) {
    catids_tmp.push_back(cat[kId]);
  }

  // only keep the categories that this process will work on 
  int catids_size = catids_tmp.size();
  int chunk_size = (catids_size + num_procs -1)/num_procs; 

  // multi-process evaluation situation:
  // (1) distribute categories across processes by chunk
  int begin = proc_id * chunk_size;
  int end = (proc_id +1) * chunk_size ;
  if (end >= catids_size ) end = catids_size ;
  catids = std::vector<int64_t>(catids_tmp.begin()+begin, catids_tmp.begin()+end);
  
  // (2) distribute categories across processes round robin
  //for (int64_t i=0; i<chunk_size; i++)
  //{
  //  if ((catids_tmp.begin()+i*num_procs+proc_id) != catids_tmp.end()) 
  //    catids.push_back(*(catids_tmp.begin()+i*num_procs+proc_id));
  //  else
  //  {
  //    catids.push_back(*catids_tmp.end());
  //    break;
  //  }
  //}

  simdjson::dom::array anns = dataset[kAnns];
  int64_t catid, category_id;

  // (3) distribute categories across processes based on gt cat-ann distribution 
  // i.e. total anns per proc are balanced 
  // Note that dt cat-ann may have a different distribution, 
  // and this distribution may have inadvertent effect on the load balance of load_res/eval across processes, and their runtime
  // Also note that when run in parallel, this distribution method requires a weighted average 
  // of the final coco_eval.stats from all the processes, with the weight being 
  // (num of categories on that process/total num of categories).
  //int64_t catids_tmp_size = catids_tmp.size();
  //std::vector<int64_t> cat_num_anns(catids_tmp_size, 0);
  //std::vector<int64_t> sum_anns_per_proc(num_procs, 0);
  //for (simdjson::dom::object annotation : anns) {
  //  category_id = annotation[kCategoryId].get_int64(); 
  //  catid = distance(catids_tmp.begin(),find(catids_tmp.begin(), catids_tmp.end(), category_id));
  //  cat_num_anns[catid]++;
  //}
  //auto cat_num_anns_inds = sort_indices(cat_num_anns, std::greater<int64_t>());
  //int argmin;
  //for (int64_t i=0; i<catids_tmp_size; i++){
  //  argmin = distance(sum_anns_per_proc.begin(), std::min_element(sum_anns_per_proc.begin(),sum_anns_per_proc.end()));
  //  if (argmin == proc_id) {
  //    catids.push_back(catids_tmp[cat_num_anns_inds[i]]);
  //  }
  //  sum_anns_per_proc[argmin] += cat_num_anns[cat_num_anns_inds[i]];
  //}
  

  gts.resize(catids.size()*imgids.size());
  simdjson::dom::element test;
  auto error = anns.at(0)[kSegm].get(test); 
  bool issegm=false;
  if (!error) 
    issegm=true;

  int64_t image_id, imgid;
  int64_t id;
  float area;
  int iscrowd;
  for (simdjson::dom::object annotation : anns) {
    category_id = annotation[kCategoryId].get_int64(); 
    if (find(catids.begin(), catids.end(), category_id) == catids.end())
      continue;
    image_id = annotation[kImageId].get_int64(); 

    id = annotation[kId].get_int64(); 
    area = static_cast<float>(annotation[kArea].get_double()); 
    iscrowd = static_cast<int>(annotation[kIsCrowd].get_int64()); 
    std::vector<float> bbox;
    for (double bb : annotation[kBbox]) {
      bbox.push_back(bb); 
    }
    
    anns_struct ann;
    if (issegm) {
      simdjson::dom::element ann_segm = annotation[kSegm];
      bool is_segm_list = ann_segm.is_array();
      bool is_cnts_list = is_segm_list ? 0 : ann_segm[kCounts].is_array();

      if (is_segm_list) {
        for (simdjson::dom::array seg1 : ann_segm) {
          std::vector<double> seg_item_l2;
          for (double seg2 : seg1) {
            seg_item_l2.push_back(seg2);
          }
          ann.segm_list.push_back(seg_item_l2);
        }
      } else if (is_cnts_list) {
        for (int64_t count : ann_segm[kCounts])
          ann.segm_counts_list.push_back(static_cast<int>(count));
        for (int64_t size : ann_segm[kSize])
          ann.segm_size.push_back(static_cast<int>(size));
      } else {
        for (int64_t size : ann_segm[kSize])
          ann.segm_size.push_back(static_cast<int>(size));
        ann.segm_counts_str = ann_segm[kCounts];
      }
    }
    
    imgid = distance(imgids.begin(),find(imgids.begin(), imgids.end(), image_id));
    catid = distance(catids.begin(),find(catids.begin(), catids.end(), category_id));
    detection_struct* tmp = &gts[catid * imgids.size() + imgid ];
    tmp->area.push_back(area);
    tmp->iscrowd.push_back(iscrowd);
    tmp->bbox.push_back(bbox);
    tmp->ignore.push_back(iscrowd!=0);
    tmp->id.push_back(id);
    int h = imgsgt[imgid].h;
    int w = imgsgt[imgid].w;
    annToRLE(ann, tmp->segm_size, tmp->segm_counts, h, w);
  }

  auto num_cats = catids.size();
  auto num_imgs = imgids.size();
  gtbboxes.resize(num_cats * num_imgs);
  gtsegm.resize(num_cats * num_imgs);

  #pragma omp parallel for schedule(guided, 8) num_threads(nthreads) collapse(2)
  for(size_t c = 0; c < num_cats; c++) {
    for(size_t i = 0; i < num_imgs; i++) {
      const auto gtsm = gts[c * imgids.size() + i];
      const auto G = gtsm.id.size();
      if(G==0)
        continue;
      
      gtsegm[c*num_imgs+i].resize(G);
      for (size_t g = 0; g < G; g++) {
        if (gtsm.segm_size[g].size()>0) {
          auto size = gtsm.segm_size[g];
          auto str = gtsm.segm_counts[g];
          char *val = const_cast<char*>(str.c_str());
          rleFrString(&(gtsegm[c*num_imgs+i][g]), val, size[0], size[1]);
          val = NULL;
        }
        for (size_t j = 0; j < gtsm.bbox[g].size(); j++)
          gtbboxes[c*num_imgs+i].push_back(static_cast<double>(gtsm.bbox[g][j]));
      }
    }
  }
  
  Py_RETURN_NONE;
}

static PyObject* cpp_load_res_numpy(PyObject* self, PyObject* args) 
{
  /* this function takes an numpy.ndarray of (rows x 7) and reads it into dts
   * the 7 columns are [image_id, category_id, bbox[0], bbox[1], bbox[2], bbox[3], score]
   * the elements need to be in dtype=numpy.float32
   * */ 

  // there must be at least 1 process to run this program
  if (num_procs<1){
    std::cout << "[cpp_load_res_numpy] Error: num_procs must be >=1" << std::endl;
    return NULL;
  }

  PyArrayObject *anns;
  // the function will use however many proesses create_index uses, 
  // and with the same (pid : data chuck id) mapping
  int nthreads;
  if (!PyArg_ParseTuple(args, "O!i|", &PyArray_Type, &anns, &nthreads)){
    std::cout << "[cpp_load_res_numpy] Error: can't parse the argument (must be numpy.ndarray)" << std::endl;
    return NULL;
  }
  int ndim = anns->nd;
  int64_t dim1 = anns->dimensions[0];
  int dim2 = anns->dimensions[1];
  if (ndim != 2 || dim2 != 7){
    std::cout << "[cpp_load_res_numpy] Error: Input array must be 2-d numpy array with 7 columns" << std::endl;
    return NULL;
  }

  if (dts.size()>0) {
    if (proc_id == 0) std::cout << "[cpp_load_res_numpy] Detection annotations already exist. Clearing vectors..." << std::endl;
    dts.clear();
  }
  dts.resize(catids.size()*imgids.size());

  float *anns_data = (float *)anns->data;

  #pragma omp parallel for num_threads(nthreads) 
  for (int64_t i = 0; i < dim1; i++) {
    int64_t image_id = static_cast<int64_t>(anns_data[i*dim2]);
    int64_t category_id = static_cast<int64_t>(anns_data[i*dim2+1]);
    if (find(catids.begin(), catids.end(), category_id) == catids.end())
      continue;
    
    std::vector<float> bbox;
    for (int d=0; d<4; d++)
      bbox.push_back(static_cast<float>(anns_data[i*dim2+d+2]));
    float score = static_cast<float>(anns_data[i*dim2+6]);
    float area = bbox[2]*bbox[3];
    int64_t id = i+1;
    int iscrowd = 0;
    
    int64_t imgid = distance(imgids.begin(), find(imgids.begin(), imgids.end(), image_id));
    int64_t catid = distance(catids.begin(), find(catids.begin(), catids.end(), category_id));

    detection_struct* tmp = &dts[catid * imgids.size() + imgid];
    tmp->area.push_back(area);
    tmp->iscrowd.push_back(iscrowd);
    tmp->bbox.push_back(bbox);
    tmp->score.push_back(score);
    tmp->id.push_back(id);
  }

  Py_RETURN_NONE;
}

static PyObject* cpp_load_res_json(PyObject* self, PyObject* args) 
{
  /* this function takes a JSON file and reads it into dts
   * the JSON file should have the following structure:
   * [
   *    {'image_id': int/int64, 'category_id': int/int64, 
   *    'segmentation': {'size': [int, int], 'counts': std::string}, 
   *    'bbox': [float, float, float, float], 'score': float/double},
   *    { ... }
   *  ]
   * the 'bbox' field is optional; the function will create it if not found
   * */ 

  // there must be at least 1 process to run this program
  if (num_procs<1){
    std::cout << "[cpp_load_res_json] Error: num_procs must be >=1" << std::endl;
    return NULL;
  }

  const char *annotation_file;
  // this function will use however many proesses create_index uses, 
  // and with the same (pid : data chuck id) mapping
  int nthreads;
  if (!PyArg_ParseTuple(args, "si|", &annotation_file, &nthreads)){
    std::cout << "[cpp_load_res_json] Error: can't parse the argument (must be a .json file)" << std::endl;
    return NULL;
  }

  if (dts.size()>0) {
    if (proc_id == 0) std::cout << "[cpp_load_res_json] Detection annotations already exist. Clearing vectors..." << std::endl;
    dts.clear();
  }
  dts.resize(catids.size()*imgids.size());

  simdjson::dom::parser parser;
  simdjson::dom::element anns= parser.load(annotation_file);

  bool iscaption = true;
  simdjson::dom::element test_caption;
  auto error_caption = anns.at(0)[kCaption].get(test_caption);
  if (error_caption) iscaption = false;

  bool isbbox=true;
  bool isbbox_exist=true;
  simdjson::dom::array test_bbox;
  auto error_bbox = anns.at(0)[kBbox].get(test_bbox);
  if (error_bbox){ 
    isbbox_exist=false;
    isbbox=false;
  }
  else{
    isbbox = isbbox_exist && (test_bbox.size() > 0);
  }
  
  bool issegm=true;
  simdjson::dom::element test_segm;
  auto error_segm = anns.at(0)[kSegm].get(test_segm);
  if (error_segm) issegm=false;
  assert(!iscaption && (isbbox||issegm));

  int64_t image_id;
  int64_t category_id;
  int64_t id=0;
  float ann_area;
  int iscrowd=0;
  float score;

  if (isbbox){
    for (simdjson::dom::object annotation : anns) {
      image_id = annotation[kImageId].get_int64(); 
      category_id = annotation[kCategoryId].get_int64(); 
      if (find(catids.begin(), catids.end(), category_id) == catids.end())
        continue;

      simdjson::dom::array ann_bbox= annotation[kBbox];
      std::vector<float> bbox;
      for (int d=0; d<4; d++)
        bbox.push_back(static_cast<float>(ann_bbox.at(d).get_double()));
      score = static_cast<float>(annotation[kScore].get_double()); 
      ann_area = bbox[2]*bbox[3];
      id++; 
      
      int64_t imgid = distance(imgids.begin(), find(imgids.begin(), imgids.end(), image_id));
      int64_t catid = distance(catids.begin(), find(catids.begin(), catids.end(), category_id));
      detection_struct* tmp = &dts[catid * imgids.size() + imgid];
      tmp->area.push_back(ann_area);
      tmp->iscrowd.push_back(iscrowd);
      tmp->bbox.push_back(bbox);
      tmp->score.push_back(score);
      tmp->id.push_back(id);
    }
  }
  else{
    for (simdjson::dom::object annotation : anns) {
      image_id = annotation[kImageId].get_int64(); 
      category_id = annotation[kCategoryId].get_int64(); 
      if (find(catids.begin(), catids.end(), category_id) == catids.end())
        continue;

      anns_struct ann;
      simdjson::dom::object ann_segm = annotation[kSegm];
      for (int64_t size : ann_segm[kSize])
        ann.segm_size.push_back(static_cast<int>(size));
      ann.segm_counts_str = ann_segm[kCounts];
      ann_area = area(ann.segm_size,ann.segm_counts_str);
      // we never use bbox in segm type in cpp ext
      //if (!isbbox_exist)
          //  ann.bbox = toBbox(ann.segm_size,ann.segm_counts_str);
      id++; 
      score = static_cast<float>(annotation[kScore].get_double()); 

      int64_t imgid = distance(imgids.begin(), find(imgids.begin(), imgids.end(), image_id));
      int64_t catid = distance(catids.begin(), find(catids.begin(), catids.end(), category_id));
      detection_struct* tmp = &dts[catid * imgids.size() + imgid];
      tmp->area.push_back(ann_area);
      tmp->iscrowd.push_back(iscrowd);
      //tmp->bbox.push_back(ann.bbox);
      tmp->score.push_back(score);
      tmp->id.push_back(id);
      auto h = imgsgt[imgid].h;
      auto w = imgsgt[imgid].w;
      annToRLE(ann,tmp->segm_size,tmp->segm_counts,h,w);
    }
  }

  Py_RETURN_NONE;
}

static PyMethodDef ext_Methods[] = {
  {"cpp_create_index", (PyCFunction)cpp_create_index, METH_VARARGS, 
    "cpp_create_index(annotation_file:str, num_procs:int, proc_id:int, nthreads:int)\n"
    "Read .json file and create ground truth map.\n"
    "Parameters: annotation_file: str \n"
    "            num_procs: int\n"
    "            proc_id: int\n"
    "            nthreads: int\n"
    "Returns:    None \n"},
  {"cpp_load_res_numpy", (PyCFunction)cpp_load_res_numpy, METH_VARARGS, 
    "cpp_load_res_numpy(results:numpy.ndarray, nthreads:int)\n"
    "Load results and create detection map.\n"
    "Parameters: results: numpy.ndarray\n"
    "            results has (number of detections in all images) rows and 7 columns,\n"
    "            and the 7 columns are [image_id, category_id, bbox[4], score];\n"
    "            results has dtype=numpy.float32.\n"
    "            nthreads:int\n"
    "Returns:    None \n"},
  {"cpp_load_res_json", (PyCFunction)cpp_load_res_json, METH_VARARGS, 
    "cpp_load_res_json(results:str, nthreads:int)\n"
    "Load results and create detection map.\n"
    "Parameters: results: str\n"
    "            results is a .json file with the following structure:\n"
    "            [\n"
    "              {'image_id': int/int64, 'category_id': int/int64,\n"
    "              'segmentation': {'size': [int, int], 'counts': std::string},\n" 
    "              'bbox': [float, float, float, float], 'score': float/double},\n"
    "              { ... }\n"
    "            ]\n"
    "            either 'bbox' or 'segmentation' needs to exist \n"
    "            nthreads:int\n"
    "Returns:    None \n"},
  {"cpp_evaluate", (PyCFunction)cpp_evaluate, METH_VARARGS, 
    "cpp_evaluate(useCats:int, areaRng:numpy.ndarray, iouThrs:numpy.ndarray, "
    "maxDets:numpy.ndarray, recThrs:numpy.ndarray, iouType:str, nthreads:int)\n "
    "Evaulate results (including the accumulation step).\n "
    "Parameters: useCats:  int\n"
    "            areaRng:  2-d numpy.ndarray (dtype=double)\n"
    "            iouThrs:  1-d numpy.ndarray (dtype=double)\n"
    "            maxDets:  1-d numpy.ndarray (dtype=int)\n"
    "            recThrs:  1-d numpy.ndarray (dtype=double)\n"
    "            iouType:  str\n"
    "            nthreads: int\n"
    "Returns:    imgids:   list\n"
    "            catids:   list\n"
    "            eval:     dict (keys=['counts','precision','recall','scores'])\n"},
  {NULL, NULL, 0, NULL} 
};

static char ext_doc[] = "COCO and COCOEval Extensions.";

static struct PyModuleDef ext_module = {
  PyModuleDef_HEAD_INIT,
  "ext",   
  ext_doc, 
  -1,         
  ext_Methods
};

PyMODINIT_FUNC
PyInit_ext(void)
{
  import_array();
  return PyModule_Create(&ext_module);
}

