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
 * RandomWoods.java
 * Copyright (C) 2005 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees;

import weka.classifiers.collective.CollectiveClassifier;
import weka.classifiers.collective.meta.CollectiveBagging;
import weka.classifiers.collective.util.Splitter;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * A modified version of the RandomForest that works in a collective way, i.e., working on labeled and unlabeled data, by using CollectiveTree instead of RandomTree.<br/>
 * <br/>
 * For more information about RandomForest, see:<br/>
 * <br/>
 * Leo Breiman (2001). Random Forests. Machine Learning. 45(1):5-32.
 * <p/>
 <!-- globalinfo-end -->
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
 * <pre> -I &lt;number of trees&gt;
 *  Number of trees to build.</pre>
 * 
 * <pre> -K &lt;number of features&gt;
 *  Number of features to consider (&lt;1=int(logM+1)).</pre>
 * 
 * <pre> -S
 *  Seed for random number generator.
 *  (default 1)</pre>
 * 
 <!-- options-end -->
 *
 * @see RandomTree
 * @see RandomForest
 * @see CollectiveTree
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */

public class RandomWoods
  extends RandomForest
  implements CollectiveClassifier {
  
  /** for serialization */
  private static final long serialVersionUID = -220198967353141592L;

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
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return  
        "A modified version of the RandomForest that works in a collective "
      + "way, i.e., working on labeled and unlabeled data, by using "
      + "CollectiveTree instead of RandomTree.\n\n"
      + "For more information about RandomForest, see:\n\n"
      + getTechnicalInformation().toString();
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
   * <pre> -I &lt;number of trees&gt;
   *  Number of trees to build.</pre>
   * 
   * <pre> -K &lt;number of features&gt;
   *  Number of features to consider (&lt;1=int(logM+1)).</pre>
   * 
   * <pre> -S
   *  Seed for random number generator.
   *  (default 1)</pre>
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
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    result.add("-folds");
    result.add("" + getSplitFolds());
    
    if (getInvertSplitFolds())
      result.add("-V");

    if (getVerbose())
      result.add("-verbose");
    
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
   * @param value the verbose state
   */
  public void setVerbose(boolean value) {
    m_Verbose = value;
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
   * @param value whether to invert the folding scheme
   */
  public void setInvertSplitFolds(boolean value) {
    m_InvertSplitFolds = value;
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

    return m_bagger.distributionForInstance(instance);
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
    Capabilities result = new Capabilities(this);

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
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
   * @see    weka.classifiers.meta.Bagging#buildClassifier(Instances)
   */
  protected void buildClassifier() throws Exception {
    m_bagger = new CollectiveBagging();
    CollectiveTree tree = new CollectiveTree();

    // set up the random tree options
    m_KValue = m_numFeatures;
    if (m_KValue < 1) 
      m_KValue = (int) Utils.log2(m_Trainset.numAttributes())+1;
    tree.setNumFeatures(m_KValue);

    // set up the bagger and build the forest
    m_bagger.setClassifier(tree);
    m_bagger.setSeed(m_randomSeed);
    m_bagger.setNumIterations(m_numTrees);
    m_bagger.setCalcOutOfBag(true);
    ((CollectiveBagging) m_bagger).buildClassifier(m_Trainset, m_Testset);
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
    m_Random          = new Random(m_randomSeed);
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
    return "Classifier............: " + this.getClass().getName() + "\n";
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  protected String toStringParameters() {
    return 
        "Tress in the forest...: " + m_numTrees + "\n"
      + "# of random feature(s): " + m_KValue + "\n";
  }
  
  /**
   * returns all the base classifiers as string representation. 
   * 
   * @return		the string representation of the model
   */
  protected String toStringModel() {
    return "Out of bag error......: " 
      + Utils.doubleToString(m_bagger.measureOutOfBagError(), 4);
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
    runClassifier(new RandomWoods(), args);
  }
}
