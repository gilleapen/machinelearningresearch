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
 * CollectiveWrapper.java
 * Copyright (C) 2005 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import weka.classifiers.Evaluation;
import weka.classifiers.collective.CollectiveRandomizableSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.RevisionUtils;

/**
 <!-- globalinfo-start -->
 * Represents a wrapper around any normal WEKA classifier.
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
 *  (default: weka.classifiers.rules.ZeroR)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.rules.ZeroR:
 * </pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author  Fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision$ 
 * @see     Evaluation
 */
public class CollectiveWrapper
  extends CollectiveRandomizableSingleClassifierEnhancer {

  /** for serialization */
  private static final long serialVersionUID = -5693670207440440346L;
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Represents a wrapper around any normal WEKA classifier.";
  }
  
  /**
   * internal function for determining the class distribution for an instance, 
   * will be overridden by derived classes. For more details about the returned 
   * array, see <code>Classifier.distributionForInstance(Instance)</code>.
   * 
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @see		weka.classifiers.Classifier#distributionForInstance(Instance)
   * @throws Exception	if something goes wrong
   */
  protected double[] getDistribution(Instance instance) throws Exception {
    return m_Classifier.distributionForInstance(instance);
  }
  
  /**
   * performs the actual building of the classifier.
   * does nothing in the wrapper.
   * 
   * @throws Exception if building fails
   */
  protected void buildClassifier() throws Exception {
    m_Classifier.buildClassifier(m_Trainset);
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
   * returns information about the classifier(s)
   * 
   * @return		information about the classifier
   */
  protected String toStringClassifier() {
    return "";
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
   * returns the best model as string representation. derived classes have to 
   * add additional information here, like printing the classifier etc.
   * 
   * @return		the string representation of the best model
   */
  protected String toStringModel() {
    return "";
  }
  
  /**
   * Returns description of the classifier.
   *
   * @return description of the classifier as a string
   */
  public String toString() {
    return m_Classifier.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.0 $");
  }
  
  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    runClassifier(new CollectiveWrapper(), args);
  }
}
