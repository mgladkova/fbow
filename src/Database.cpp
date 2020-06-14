#include "Database.h"

namespace fbow {

// --------------------------------------------------------------------------


Database::Database
  (bool use_di, int di_levels)
  : m_voc(nullptr), m_use_di(use_di), m_dilevels(di_levels), m_nentries(0)
{
}

// --------------------------------------------------------------------------

Database::Database
  (const Vocabulary &voc, bool use_di, int di_levels)
  : m_voc(nullptr), m_use_di(use_di), m_dilevels(di_levels)
{
  setVocabulary(voc);
  clear();
}

// --------------------------------------------------------------------------


Database::Database
  (const Database &db)
  : m_voc(nullptr)
{
  *this = db;
}

// --------------------------------------------------------------------------


Database::Database
  (const std::string &filename)
  : m_voc(nullptr)
{
  load(filename);
}

// --------------------------------------------------------------------------


Database::Database
  (const char *filename)
  : m_voc(nullptr)
{
  load(filename);
}

// --------------------------------------------------------------------------


Database::~Database(void){
  m_voc->clear();
}

// --------------------------------------------------------------------------


Database& Database::operator=
  (const Database &db)
{
  if(this != &db)
  {
    m_dfile = db.m_dfile;
    m_dilevels = db.m_dilevels;
    m_ifile = db.m_ifile;
    m_nentries = db.m_nentries;
    m_use_di = db.m_use_di;
    if (db.m_voc != nullptr) setVocabulary(*db.m_voc);
  }
  return *this;
}

// --------------------------------------------------------------------------

EntryId Database::add(
  const  cv::Mat &features,
  BowVector *bowvec, FeatureVector *fvec)
{
    std::vector<cv::Mat> vf(features.rows);
    for(int r=0;r<features.rows;r++) vf[r]=features.rowRange(r,r+1);
    return add(vf,bowvec,fvec);
}

EntryId Database::add(
  const std::vector<cv::Mat> &features,
  BowVector *bowvec, FeatureVector *fvec)
{
  BowVector aux;
  BowVector& v = (bowvec ? *bowvec : aux);

  if(m_use_di && fvec != NULL)
  {
    m_voc->transform(features, v, *fvec, m_dilevels); // with features
    return add(v, *fvec);
  }
  else if(m_use_di)
  {
    FeatureVector fv;
    m_voc->transform(features, v, fv, m_dilevels); // with features
    return add(v, fv);
  }
  else if(fvec != NULL)
  {
    m_voc->transform(features, v, *fvec, m_dilevels); // with features
    return add(v);
  }
  else
  {
    m_voc->transform(features, v); // with features
    return add(v);
  }
}

// ---------------------------------------------------------------------------


EntryId Database::add(const BowVector &v,
  const FeatureVector &fv)
{
  EntryId entry_id = m_nentries++;

  BowVector::const_iterator vit;
  std::vector<unsigned int>::const_iterator iit;

  if(m_use_di){
    // update direct file
    if(entry_id == m_dfile.size())
    {
      m_dfile.push_back(fv);
    }
    else
    {
      m_dfile[entry_id] = fv;
    }
  }

  // update inverted file
  for(vit = v.begin(); vit != v.end(); ++vit){
    const WordId& word_id = vit->first;
    const WordValue& word_weight = vit->second;

    if (word_id >= m_ifile.size()){
      continue;
    }

    IFRow& ifrow = m_ifile[word_id];
    ifrow.push_back(IFPair(entry_id, word_weight));
  }

  return entry_id;
}

// --------------------------------------------------------------------------


  void Database::setVocabulary
  (const Vocabulary& voc)
{
  m_voc.reset();
  m_voc = std::shared_ptr<Vocabulary>(new Vocabulary(voc));
  clear();
}

// --------------------------------------------------------------------------


  void Database::setVocabulary
  (const Vocabulary& voc, bool use_di, int di_levels)
{
  m_use_di = use_di;
  m_dilevels = di_levels;
  m_voc.reset();
  m_voc = std::shared_ptr<Vocabulary>(new Vocabulary(voc));
  clear();
}

// --------------------------------------------------------------------------


const std::shared_ptr<Vocabulary> Database::getVocabulary() const {
  return m_voc;
}

// --------------------------------------------------------------------------


void Database::clear(){
  // resize vectors
  m_ifile.resize(0);
  m_ifile.resize(m_voc->size());
  m_dfile.resize(0);
  m_nentries = 0;
}

// --------------------------------------------------------------------------


void Database::allocate(int nd, int ni)
{
  // m_ifile already contains |words| items
  if(ni > 0)
  {
    for(auto rit = m_ifile.begin(); rit != m_ifile.end(); ++rit)
    {
      int n = (int)rit->size();
      if(ni > n)
      {
        rit->resize(ni);
        rit->resize(n);
      }
    }
  }

  if(m_use_di && (int)m_dfile.size() < nd)
  {
    m_dfile.resize(nd);
  }
}



// --------------------------------------------------------------------------

void Database::query(
  const  cv::Mat &features,
  QueryResults &ret, int max_results, int max_id) const
{
    BowVector vec;
    m_voc->transform(features, vec);
    query(vec, ret, max_results, max_id);
}



void Database::query(
  const std::vector<cv::Mat> &features,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector vec;
  m_voc->transform(features, vec);
  query(vec, ret, max_results, max_id);
}

// --------------------------------------------------------------------------


void Database::query(const BowVector &vec,
                    QueryResults &ret, int max_results, int max_id) const
{
  ret.resize(0);

  switch(m_voc->getScoringType())
  {
    case L1_NORM:
      queryL1(vec, ret, max_results, max_id);
      break;

    case L2_NORM:
      queryL2(vec, ret, max_results, max_id);
      break;

    default:
      std::cerr << "Other scoring types are not supported :(" << std::endl;
  }
}

// --------------------------------------------------------------------------


void Database::queryL1(const BowVector &vec,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;

  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;

  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;

    const IFRow& row = m_ifile[word_id];

    // IFRows are sorted in ascending entry_id order

    for(auto rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;

      if((int)entry_id < max_id || max_id == -1)
      {
        double value = fabs(qvalue - dvalue) - fabs(qvalue) - fabs(dvalue);

        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value;
        }
        else
        {
          pairs.insert(pit,
            std::map<EntryId, double>::value_type(entry_id, value));
        }
      }

    } // for each inverted row
  } // for each query word

  // move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    ret.push_back(Result(pit->first, pit->second));
  }

  // resulting "scores" are now in [-2 best .. 0 worst]

  // sort vector in ascending order of score
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // complete and scale score to [0 worst .. 1 best]
  // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|)
  //		for all i | v_i != 0 and w_i != 0
  // (Nister, 2006)
  // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
  QueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++)
    qit->Score = -qit->Score/2.0;
}

// --------------------------------------------------------------------------


void Database::queryL2(const BowVector &vec,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;

  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;

  //map<EntryId, int> counters;
  //map<EntryId, int>::iterator cit;

  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;

    const IFRow& row = m_ifile[word_id];

    // IFRows are sorted in ascending entry_id order

    //std::cout << "Word id = " << word_id << std::endl;

    if (word_id >= m_voc->size()){
      continue;
    }

    for(auto rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;

      if((int)entry_id < max_id || max_id == -1)
      {
        double value = - qvalue * dvalue; // minus sign for sorting trick

        pit = pairs.lower_bound(entry_id);
        //cit = counters.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value;
          //cit->second += 1;
        }
        else
        {
          pairs.insert(pit,
            std::map<EntryId, double>::value_type(entry_id, value));

          //counters.insert(cit,
          //  map<EntryId, int>::value_type(entry_id, 1));
        }
      }

    } // for each inverted row
  } // for each query word

  // move to vector
  ret.reserve(pairs.size());
  //cit = counters.begin();
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)//, ++cit)
  {
    ret.push_back(Result(pit->first, pit->second));// / cit->second));
  }

  // resulting "scores" are now in [-1 best .. 0 worst]

  // sort vector in ascending order of score
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // complete and scale score to [0 worst .. 1 best]
  // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i)
    //		for all i | v_i != 0 and w_i != 0 )
    // (Nister, 2006)
    QueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++)
  {
    if(qit->Score <= -1.0) // rounding error
      qit->Score = 1.0;
    else
      qit->Score = 1.0 - sqrt(1.0 + qit->Score); // [0..1]
      // the + sign is ok, it is due to - sign in
      // value = - qvalue * dvalue
  }

}

const FeatureVector& Database::retrieveFeatures
  (EntryId id) const
{
  assert(id < size());
  return m_dfile[id];
}

// --------------------------------------------------------------------------


void Database::save(const std::string &filename) const {
  m_voc->saveToFile(filename);
}

// --------------------------------------------------------------------------


void Database::load(const std::string &filename){
  // load voc first
  // subclasses must instantiate m_voc before calling this ::load
  if(m_voc == nullptr){
    m_voc = std::shared_ptr<Vocabulary>(new Vocabulary());
  }
  m_voc->readFromFile(filename);
}

// --------------------------------------------------------------------------



std::ostream& operator<<(std::ostream &os, const Database &db){
  os << "Database: Entries = " << db.size() << ", "
    "Using direct index = " << (db.usingDirectIndex() ? "yes" : "no");

  if(db.usingDirectIndex())
    os << ", Direct index levels = " << db.getDirectIndexLevels();

  auto voc = db.getVocabulary();
  voc->toStream(os);

  return os;
}

}
