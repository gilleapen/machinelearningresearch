/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * CollectiveBagging.java
 * Copyright (C) 2005 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import weka.classifiers.Classifier;
import weka.classifiers.collective.CollectiveClassifier;
import weka.classifiers.collective.util.Splitter;
import weka.classifiers.meta.Bagging;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * Class for bagging a collective classifier to reduce variance. Can do classification and regression depending on the base learner.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Leo Breiman (1996). Bagging predictors. Machine Learning. 24(2):123-140.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman1996,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {2},
 *    pages = {123-140},
 *    title = {Bagging predictors},
 *    volume = {24},
 *    year = {1996}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -folds &lt;folds&gt;
 *  The number of folds for splitting the training set into
 *  train and test set. The first fold is always the training
 *  set. With '-V' you can invert this, i.e., instead of 20/80
 *  for 5 folds you'll get 80/20.
 *  (default 5)</pre>
 * 
 * <pre> -V
 *  Inverts the fold selection, i.e., instead of using the first
 *  fold for the training set it is used for test set and the
 *  remaining folds for training.</pre>
 * 
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 * 
 * <pre> -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)</pre>
 * 
 * <pre> -O
 *  Calculate the out of bag error.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.collective.meta.SimpleCollective)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.collective.meta.SimpleCollective:
 * </pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -R &lt;num&gt;
 *  Number of restarts.
 *  (default 10)</pre>
 * 
 * <pre> -log
 *  Creates logs in the tmp directory for all kinds of internal data.
 *  Use only for debugging purposes!
 * </pre>
 * 
 * <pre> -U
 *  Updates also the labels of the training set.
 * </pre>
 * 
 * <pre> -eval &lt;num&gt;
 *  The type of evaluation to use (0 = Randomwalk/Last model used for 
 *  prediction, 1=Randomwalk/Best model used for prediction,
 *  2=Hillclimbing).
 * </pre>
 * 
 * <pre> -compare &lt;num&gt;
 *  The type of comparisong used for comparing models.
 *  (0=overall RMS, 1=RMS on train set, 2=RMS on test set, 
 *  3=Accuracy on train set)
 * </pre>
 * 
 * <pre> -flipper "&lt;classname [parameters]&gt;"
 *  The flipping algorithm (and optional parameters) to use for 
 *  flipping labels.
 * </pre>
 * 
 * <pre> -folds &lt;folds&gt;
 *  The number of folds for splitting the training set into
 *  train and test set. The first fold is always the training
 *  set. With '-V' you can invert this, i.e., instead of 20/80
 *  for 5 folds you'll get 80/20.
 *  (default 5)</pre>
 * 
 * <pre> -V
 *  Inverts the fold selection, i.e., instead of using the first
 *  fold for the training set it is used for test set and the
 *  remaining folds for training.</pre>
 * 
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 * 
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.J48)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.J48:
 * </pre>
 * 
 * <pre> -U
 *  Use unpruned tree.</pre>
 * 
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 * 
 * <pre> -R
 *  Use reduced error pruning.</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)</pre>
 * 
 * <pre> -B
 *  Use binary splits only.</pre>
 * 
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 * 
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 * 
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 * 
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p/>
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */

public class CollectiveBagging
  extends Bagging
  implements CollectiveClassifier {
  
  /** for serialization */
  private static final long serialVersionUID = 2984501645615597451L;

  /** whether the classifier was already built
   * (<code>buildClassifier(Instances)</code> only stores the training set.
   * Actual training is done in
   * <code>distributionForInstance(Instances)</code>)
   */
  protected boolean m_ClassifierBuilt = false;
  
  /** whether to output some information during improving the classifier */
  protected boolean m_Verbose = false;

  /** The test instances */
  protected Instances m_Testset = null;

  /** The test instances (with the original labels) 
   * @see #m_UseInsight */
  protected Instances m_TestsetOriginal = null;
  
  /** The training instances */
  protected Instances m_Trainset = null;
  
  /** Random number generator */
  protected Random m_Random = null;
  
  /** The number of folds to split the training set into train and test set.
   *  E.g. 5 folds result in 20% train and 80% test set. */
  protected int m_SplitFolds = 0;
  
  /** Whether to invert the folds, i.e., instead of taking the first fold as
   *  training set it is taken as test set and the rest as training. */
  protected boolean m_InvertSplitFolds = false;

  /** Stores the original labels of the dataset. Used for outputting some more 
   * statistics about the learning process. */
  protected boolean m_UseInsight = false;

  /**
   * initializes the classifier
   */
  public CollectiveBagging() {
    super();
    m_Classifier = new SimpleCollective();
  }
    
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Class for bagging a collective classifier to reduce variance. Can do classification "
      + "and regression depending on the base learner.\n\n"
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * String describing default classifier.
   * 
   * @return		the classname
   */
  protected String defaultClassifierString() {
    return SimpleCollective.class.getName();
  }
  
  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector result = new Vector();
    
    result.addElement(new Option(
        "\tThe number of folds for splitting the training set into\n"
        + "\ttrain and test set. The first fold is always the training\n"
        + "\tset. With '-V' you can invert this, i.e., instead of 20/80\n"
        + "\tfor 5 folds you'll get 80/20.\n"
        + "\t(default 0 - no splitting, test = train)",
        "folds", 1, "-folds <folds>"));
    
    result.addElement(new Option(
        "\tInverts the fold selection, i.e., instead of using the first\n"
        + "\tfold for the training set it is used for test set and the\n"
        + "\tremaining folds for training.",
        "V", 0, "-V"));
    
    result.addElement(new Option(
        "\tWhether to print some more information during building the\n"
        + "\tclassifier.\n"
        + "\t(default is off)",
        "verbose", 0, "-verbose"));
    
    Enumeration en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());

    return result.elements();
  }
  
  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -folds &lt;folds&gt;
   *  The number of folds for splitting the training set into
   *  train and test set. The first fold is always the training
   *  set. With '-V' you can invert this, i.e., instead of 20/80
   *  for 5 folds you'll get 80/20.
   *  (default 5)</pre>
   * 
   * <pre> -V
   *  Inverts the fold selection, i.e., instead of using the first
   *  fold for the training set it is used for test set and the
   *  remaining folds for training.</pre>
   * 
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   * 
   * <pre> -P
   *  Size of each bag, as a percentage of the
   *  training set size. (default 100)</pre>
   * 
   * <pre> -O
   *  Calculate the out of bag error.</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.collective.meta.SimpleCollective)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.collective.meta.SimpleCollective:
   * </pre>
   * 
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   * 
   * <pre> -R &lt;num&gt;
   *  Number of restarts.
   *  (default 10)</pre>
   * 
   * <pre> -log
   *  Creates logs in the tmp directory for all kinds of internal data.
   *  Use only for debugging purposes!
   * </pre>
   * 
   * <pre> -U
   *  Updates also the labels of the training set.
   * </pre>
   * 
   * <pre> -eval &lt;num&gt;
   *  The type of evaluation to use (0 = Randomwalk/Last model used for 
   *  prediction, 1=Randomwalk/Best model used for prediction,
   *  2=Hillclimbing).
   * </pre>
   * 
   * <pre> -compare &lt;num&gt;
   *  The type of comparisong used for comparing models.
   *  (0=overall RMS, 1=RMS on train set, 2=RMS on test set, 
   *  3=Accuracy on train set)
   * </pre>
   * 
   * <pre> -flipper "&lt;classname [parameters]&gt;"
   *  The flipping algorithm (and optional parameters) to use for 
   *  flipping labels.
   * </pre>
   * 
   * <pre> -folds &lt;folds&gt;
   *  The number of folds for splitting the training set into
   *  train and test set. The first fold is always the training
   *  set. With '-V' you can invert this, i.e., instead of 20/80
   *  for 5 folds you'll get 80/20.
   *  (default 5)</pre>
   * 
   * <pre> -V
   *  Inverts the fold selection, i.e., instead of using the first
   *  fold for the training set it is used for test set and the
   *  remaining folds for training.</pre>
   * 
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   * 
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.J48)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.trees.J48:
   * </pre>
   * 
   * <pre> -U
   *  Use unpruned tree.</pre>
   * 
   * <pre> -C &lt;pruning confidence&gt;
   *  Set confidence threshold for pruning.
   *  (default 0.25)</pre>
   * 
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf.
   *  (default 2)</pre>
   * 
   * <pre> -R
   *  Use reduced error pruning.</pre>
   * 
   * <pre> -N &lt;number of folds&gt;
   *  Set number of folds for reduced error
   *  pruning. One fold is used as pruning set.
   *  (default 3)</pre>
   * 
   * <pre> -B
   *  Use binary splits only.</pre>
   * 
   * <pre> -S
   *  Don't perform subtree raising.</pre>
   * 
   * <pre> -L
   *  Do not clean up after the tree has been built.</pre>
   * 
   * <pre> -A
   *  Laplace smoothing for predicted probabilities.</pre>
   * 
   * <pre> -Q &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p/>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String        tmpStr;

    tmpStr = Utils.getOption("folds", options);
    if (tmpStr.length() != 0)
      setSplitFolds(Integer.parseInt(tmpStr));
    else
      setSplitFolds(0);
    
    setInvertSplitFolds(Utils.getFlag('V', options));
 
    setVerbose(Utils.getFlag("verbose", options));
    
    super.setOptions(options);
  }
  
  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result  = new Vector();

    result.add("-folds");
    result.add("" + getSplitFolds());
    
    if (getInvertSplitFolds())
      result.add("-V");

    if (getVerbose())
      result.add("-verbose");
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }
  
  /**
   * sets the instances used for testing
   * 
   * @param value the instances used for testing
   */
  public void setTestSet(Instances value) {
    m_ClassifierBuilt = false;
    m_Testset         = value;
    if (getUseInsight())
      m_TestsetOriginal = new Instances(value);
    else
      m_TestsetOriginal = null;
  }
  
  /**
   * returns the Test Set
   *
   * @return the Test Set
   */
  public Instances getTestSet() {
    return m_Testset;
  }
  
  /**
   * returns the Training Set 
   *
   * @return the Training Set
   */
  public Instances getTrainingSet() {
    return m_Trainset;
  }
  
  /**
   * Set the verbose state.
   *
   * @param verbose the verbose state
   */
  public void setVerbose(boolean verbose) {
    m_Verbose = verbose;
  }
  
  /**
   * Gets the verbose state
   *
   * @return the verbose state
   */
  public boolean getVerbose() {
    return m_Verbose;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String verboseTipText() {
    return "Whether to ouput some additional information during building.";
  }
  
  /**
   * Set the percentage for splitting the train set into train and test set.
   * Use 0 for no splitting, which results in test = train.
   *
   * @param value the split percentage (1/splitFolds*100)
   */
  public void setSplitFolds(int value) {
    if (value >= 2)
      m_SplitFolds = value;
    else
      m_SplitFolds = 0;
  }
  
  /**
   * Gets the split percentage for splitting train set into train and test set
   *
   * @return the split percentage (1/splitFolds*100)
   */
  public int getSplitFolds() {
    return m_SplitFolds;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String splitFoldsTipText() {
    return "The percentage to use for splitting the train set into train and test set.";
  }
  
  /**
   * Sets whether to use the first fold as training set (= FALSE) or as
   * test set (= TRUE).
   *
   * @param invertSplitFolds whether to invert the folding scheme
   */
  public void setInvertSplitFolds(boolean invertSplitFolds) {
    m_InvertSplitFolds = invertSplitFolds;
  }
  
  /**
   * Gets whether to use the first fold as training set (= FALSE) or as
   * test set (= TRUE)
   *
   * @return whether to invert the folding scheme
   */
  public boolean getInvertSplitFolds() {
    return m_InvertSplitFolds;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String invertSplitFoldsTipText() {
    return "Whether to invert the folding scheme.";
  }

  /**
   * Whether to use the labels of the test set to output more statistics
   * (not used for learning and only for debugging purposes)
   * 
   * @param value	if true, more statistics are output
   */
  public void setUseInsight(boolean value) {
    m_UseInsight = value;
  }

  /**
   * Returns whether we use the labels of the test set to output some more
   * statistics.
   * 
   * @return		true if more statistics are output
   */
  public boolean getUseInsight() {
    return m_UseInsight;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String useInsightTipText() {
    return "Whether to use the original labels of the test set to generate more statistics (not used for learning!).";
  }
  
  /**
   * resets the classifier
   */
  public void reset() {
    if (getDebug() || getVerbose())
      System.out.println("Reseting the classifier...");
    m_ClassifierBuilt = false;
  }
  
  /**
   * splits the train set into train and test set if no test set was provided,
   * according to the set SplitFolds.
   *
   * @see #getSplitFolds()
   * @see #getInvertSplitFolds()
   * @throws Exception if anything goes wrong with the Filter
   */
  protected void splitTrainSet() throws Exception {
    Splitter        splitter;

    splitter = new Splitter(m_Trainset);
    splitter.setSplitFolds(getSplitFolds());
    splitter.setInvertSplitFolds(getInvertSplitFolds());
    splitter.setVerbose(getVerbose());
    
    m_Trainset = splitter.getTrainset();
    m_Testset  = splitter.getTestset();
  }
  
  /**
   * checks whether the classifier was build and if not performs the build (but 
   * only if the testset is not <code>null</code>).
   * 
   * @see 		#m_ClassifierBuilt 
   * @return		true if the classifier was built
   */
  protected boolean checkBuiltStatus() {
    boolean result = m_ClassifierBuilt;
    
    if ( (!result) && (m_Testset != null) ) {
      try {
        buildClassifier(m_Trainset, m_Testset);
        result = m_ClassifierBuilt;
      }
      catch (Exception e) {
        throw new IllegalStateException(e);
      }
    }
    
    return result;
  }
  
  /**
   * Predicts the class memberships for a given instance. If
   * an instance is unclassified, the returned array elements
   * must be all zero. If the class is numeric, the array
   * must consist of only one element, which contains the
   * predicted value. Note that a classifier MUST implement
   * either this or classifyInstance().<br/>
   * Note: if a derived class should override this method, make
   * sure it calls <code>checkBuiltStatus()</code>.
   *
   * @param instance      the instance to be classified
   * @return              an array containing the estimated membership 
   *                      probabilities of the test instance in each class 
   *                      or the numeric prediction
   * @throws Exception if distribution could not be 
   *                      computed successfully
   * @see #checkBuiltStatus()
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
    // no testset if after buildClassifier(Instances) call 
    // this method is -> split training set
    // e.g., in the Explorer ("Classify")
    if (m_Testset == null) {
      splitTrainSet();
      generateSets();
    }
   
    checkBuiltStatus();
    
    return getDistribution(instance); 
  }
  
  /**
   * internal function for determining the class distribution for an instance, 
   * will be overridden by derived classes. For more details about the returned 
   * array, see <code>Classifier.distributionForInstance(Instance)</code>.
   * 
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @see 		weka.classifiers.Classifier#distributionForInstance(Instance)
   * @throws Exception	if something goes wrong
   */
  protected double[] getDistribution(Instance instance) 
    throws Exception {

    double [] sums = new double [instance.numClasses()], newProbs; 
    
    for (int i = 0; i < m_NumIterations; i++) {
      if (instance.classAttribute().isNumeric() == true) {
	sums[0] += m_Classifiers[i].classifyInstance(instance);
      } else {
	newProbs = m_Classifiers[i].distributionForInstance(instance);
	for (int j = 0; j < newProbs.length; j++)
	  sums[j] += newProbs[j];
      }
    }
    if (instance.classAttribute().isNumeric() == true) {
      sums[0] /= (double)m_NumIterations;
      return sums;
    } else if (Utils.eq(Utils.sum(sums), 0)) {
      return sums;
    } else {
      Utils.normalize(sums);
      return sums;
    }
  }

  /**
   * Checks the data, whether it can be used. If not Exceptions are thrown
   * @throws Exception if the data doesn't fit in any way
   */
  protected void checkData() throws Exception {
    if (m_Testset == null)
      throw new Exception("No Test instances provided!");
    
    if (!m_Trainset.equalHeaders(m_Testset))
      throw new Exception("Training and Test Set not compatible!");
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.disableAllClassDependencies();
    result.enable(Capability.BINARY_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    
    return result;
  }
  
  /**
   * the standard collective classifier accepts only nominal, binary classes
   * otherwise an exception is thrown
   * @throws Exception if the data doesn't have a nominal, binary class
   */
  protected void checkDataRestrictions() throws Exception {
    // can classifier handle the data?
    getCapabilities().testWithFail(m_Trainset);

    // remove instances with missing class
    m_Trainset = new Instances(m_Trainset);
    m_Trainset.deleteWithMissingClass();
    if (m_Testset != null)
      m_Testset = new Instances(m_Testset);
  }
  
  /**
   * removes all the labels from the test set
   * 
   * @throws Exception if anything goes wrong
   */
  protected void generateSets() throws Exception {
    Instances         instances;
    Instance          instance;
    
    instances = new Instances(m_Testset, 0);

    for (int i = 0; i < m_Testset.numInstances(); i++) {
      instance = (Instance) m_Testset.instance(i).copy();
      instance.setClassMissing();
      instances.add(instance);
    }

    m_Testset = instances;
  }
  
  /**
   * Performs the actual building of the classifier. Based on the
   * <code>buildClassifier(Instances)</code> method in the Bagging class.
   * 
   * @throws Exception if building fails
   * @see    Bagging#buildClassifier(Instances)
   */
  protected void buildClassifier() throws Exception {
    if (m_CalcOutOfBag && (m_BagSizePercent != 100))
      throw new IllegalArgumentException("Bag size needs to be 100% if " +
					 "out-of-bag error is to be calculated!");
    if (!(m_Classifier instanceof CollectiveClassifier))
      throw new Exception("Classifier must be a CollectiveClassifier!");

    double outOfBagCount = 0.0;
    double errorSum = 0.0;

    m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);

    int bagSize = m_Trainset.numInstances() * m_BagSizePercent / 100;
    Random random = new Random(m_Seed);
    for (int j = 0; j < m_Classifiers.length; j++) {
      Instances bagData = null;
      boolean[] inBag = null;
      // create the in-bag dataset
      if (m_CalcOutOfBag) {
	inBag = new boolean[m_Trainset.numInstances()];
	bagData = resampleWithWeights(m_Trainset, random, inBag);
      } 
      else {
	bagData = m_Trainset.resampleWithWeights(random);
	if (bagSize < m_Trainset.numInstances()) {
	  bagData.randomize(random);
	  Instances newBagData = new Instances(bagData, 0, bagSize);
	  bagData = newBagData;
	}
      }
      if (m_Classifier instanceof Randomizable)
	((Randomizable) m_Classifiers[j]).setSeed(random.nextInt());

      // build the classifier
      ((CollectiveClassifier) m_Classifiers[j]).buildClassifier(
                                                        bagData, m_Testset);
      if (m_CalcOutOfBag) {
	// calculate out of bag error
	for (int i = 0; i < inBag.length; i++) {  
	  if (!inBag[i]) {
	    Instance outOfBagInst = m_Trainset.instance(i);
	    outOfBagCount += outOfBagInst.weight();
	    if (m_Trainset.classAttribute().isNumeric()) {
	      errorSum += outOfBagInst.weight() *
		Math.abs(m_Classifiers[j].classifyInstance(outOfBagInst)
			 - outOfBagInst.classValue());
	    } 
            else {
	      if (m_Classifiers[j].classifyInstance(outOfBagInst)
		  != outOfBagInst.classValue()) {
		errorSum += outOfBagInst.weight();
	      }
	    }
	  }
	}
      }
    }
    m_OutOfBagError = errorSum / outOfBagCount;
  }

  /**
   * here initialization and building, possible iterations will happen
   * 
   * @throws Exception	if something goes wrong
   */
  protected void build() throws Exception {
    buildClassifier();
  }
  
  /**
   * Method for building this classifier. Since the collective classifiers
   * also need the test set, we only store here the training set.  
   * 
   * @param training        the training set to use
   * @throws Exception      derived classes may throw Exceptions
   */
  public void buildClassifier(Instances training) throws Exception {
    m_ClassifierBuilt = false;
    m_Trainset        = training;

    // set class index?
    if (m_Trainset.classIndex() == -1)
      m_Trainset.setClassIndex(m_Trainset.numAttributes() - 1);

    // necessary for JUnit tests
    checkDataRestrictions();
  }
  
  /**
   * Method for building this classifier.
   * 
   * @param training	the training instances
   * @param test	the test instances
   * @throws Exception	if something goes wrong
   */
  public void buildClassifier(Instances training, Instances test) throws Exception {
    m_ClassifierBuilt = true;
    m_Random          = new Random(m_Seed);
    m_Trainset        = training;
    m_Testset         = test;

    // set class index?
    if ( (m_Trainset.classIndex() == -1) || (m_Testset.classIndex() == -1) ) {
      m_Trainset.setClassIndex(m_Trainset.numAttributes() - 1);
      m_Testset.setClassIndex(m_Trainset.numAttributes() - 1);
    }

    // are datasets correct?
    checkData();

    // any other data restrictions not met?
    checkDataRestrictions();
    
    // generate sets
    generateSets();
    
    // performs the restarts/iterations
    build();
    
    m_Random = null;
  }
  
  /**
   * returns information about the classifier(s)
   * 
   * @return		information about the classifier
   */
  protected String toStringClassifier() {
    return "Classifier............: " + m_Classifier.getClass().getName() + "\n";
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  protected String toStringParameters() {
    return "";
  }
  
  /**
   * returns all the base classifiers as string representation. 
   * 
   * @return		the string representation of the best model
   */
  protected String toStringModel() {
    StringBuffer        text;
    int                 i;

    text = new StringBuffer();
    text.append("All the base classifiers: \n\n");
    for (i = 0; i < m_Classifiers.length; i++)
      text.append(m_Classifiers[i].toString() + "\n\n");
    
    if (m_CalcOutOfBag)
      text.append("Out of bag error: "
		  + Utils.doubleToString(m_OutOfBagError, 4)
		  + "\n\n");

    return text.toString();
  }
  
  /**
   * Returns description of the classifier.<br/>
   * Note: if a derived class overrides this method, make sure it calls
   * <code>checkBuiltStatus()</code> to assure a model has been built.
   *
   * @return description of the classifier as a string
   * @see #checkBuiltStatus()
   */
  public String toString() {
    StringBuffer    text;
    String          classifier;
    
    text       = new StringBuffer();
    classifier = getClass().getName().replaceAll(".*\\.", "");
    
    text.append(classifier + "\n" + classifier.replaceAll(".", "-"));
    text.append("\n\n");
    text.append(toStringClassifier());
    text.append(toStringParameters());
    text.append("\n");
    if (!checkBuiltStatus())
      text.append("No Test set provided so far, hence no model built yet! See below for model...");
    else
      text.append(toStringModel());
    
    return text.toString();
  }

  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    runClassifier(new CollectiveBagging(), args);
  }
}
