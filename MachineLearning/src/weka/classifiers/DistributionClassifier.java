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
 *    DistributionClassifier.java
 *    Copyright (C) 1999 Eibe Frank, Len Trigg
 *    Modified by Prem Melville
 */

package weka.classifiers;

import weka.core.*;

/** 
 * Abstract classification model that produces (for each test instance)
 * an estimate of the membership in each class 
 * (ie. a probability distribution).
 *
 * @author   Eibe Frank (trigg@cs.waikato.ac.nz)
 * @author   Len Trigg (trigg@cs.waikato.ac.nz)
 * @author   Prem Melville (melville@cs.utexas.edu)
 * @version $Revision: 1.5 $
 */
public abstract class DistributionClassifier extends Classifier {

  /**
   * Predicts the class memberships for a given instance. If
   * an instance is unclassified, the returned array elements
   * must be all zero. If the class is numeric, the array
   * must consist of only one element, which contains the
   * predicted value.
   *
   * @param instance the instance to be classified
   * @return an array containing the estimated membership 
   * probabilities of the test instance in each class (this 
   * should sum to at most 1)
   * @exception Exception if distribution could not be 
   * computed successfully
   */
  public abstract double[] distributionForInstance(Instance instance) 
       throws Exception;

    /*
     * Calculates the entropy of this classifier on the given instance.
     */
    public double calculateEntropy(Instance instance) throws Exception{
	double []probs = distributionForInstance(instance);
	return(ContingencyTables.entropy(probs));
    }
    
    /*
     * Calculates a score that is a hybrid of margins and uncertainty
     */
    public double calculateHybridScore(Instance instance) throws Exception{
	//Prefer examples that are "near" misses, then those that are
	//certain misses, and lastly those that are just uncertain.
	double score;
	int pred = (int) classifyInstance(instance);
	int classIndex = (int) instance.classValue();//assuming the class is nominal 
	
	if(pred == classIndex){//correctly classified
	    score = calculateMargin(instance) + 1;
	}else{//incorrectly classified
	    score = Math.abs(calculateLabeledInstanceMargin(instance)) - 1;
	}
	
	assert ((-1.001 <= score) && (score <= 2.001)) : "Score out of range!"+score;
	return(score);
    }

    /*
     * Calculates a score that randomly selects the mispredictions and then prefers uncertainity
     */
    public double calculateRandomHybridScore(Instance instance) throws Exception{
	double score;
	int pred = (int) classifyInstance(instance);
	int classIndex = (int) instance.classValue();//assuming the class is nominal 
	
	if(pred == classIndex){//correctly classified
	    score = calculateMargin(instance);
	}else{//incorrectly classified
	    score = -1.0;//assuming the instances have been randomized in the training sets
	    //leads to random selection of incorrectly classified examples
	}
	assert ((-1.001 <= score) && (score <= 1.001)) : "Score out of range! "+score;
	return(score);
    }
    
    /*
     * Calculates the (normalized) margin on this classifier on the given instance.
     */
    public double calculateMargin(Instance instance) throws Exception{
	//The margin is calculated as the difference in the posterior
	//probability of the most popular class and the second most
	//popular class
	double []probs = distributionForInstance(instance);
	int maxIndex = Utils.maxIndex(probs); 
	double secondMax = 0;
	
	for (int i = 0; i < probs.length; i++) {
	    if ((probs[i] > secondMax) && (i != maxIndex)) {
		secondMax = probs[i];
	    }
	}
	assert (probs[maxIndex] - secondMax) >= 0 : "Error in margin calculation.";
	return(probs[maxIndex] - secondMax);
    }
    

    /*
     * Calculates the (normalized) margin of this classifier on the given LABELED instance.
     * Only applicable to nominal classes.
     */
    public double calculateLabeledInstanceMargin(Instance instance) throws Exception{
	//The margin is calculated as the difference in the posterior
	//probability of the most popular class and the second most
	//popular class
	double []probs = distributionForInstance(instance);
	int classIndex = (int) instance.classValue();//assuming the class is nominal 
	double maxIncorrect=0;
	
	for (int i = 0; i < probs.length; i++) {
	    if ((probs[i] > maxIncorrect) && (i != classIndex)) {
		maxIncorrect = probs[i];
	    }
	}
	double margin = probs[classIndex] - maxIncorrect;
	assert ((-1.001 <= margin) && (margin <= 1.001)) : "Margin out of range!"+margin;
	return margin;
    }


  /**
   * Classifies the given test instance. The instance has to belong to a
   * dataset when it's being classified.
   *
   * @param instance the instance to be classified
   * @return the predicted most likely class for the instance or 
   * Instance.missingValue() if no prediction is made
   * @exception Exception if an error occurred during the prediction
   */
  public double classifyInstance(Instance instance) throws Exception {

    double [] dist = distributionForInstance(instance);
    if (dist == null) {
      throw new Exception("Null distribution predicted");
    }
    switch (instance.classAttribute().type()) {
    case Attribute.NOMINAL:
      double max = 0;
      int maxIndex = 0;
      
      for (int i = 0; i < dist.length; i++) {
	if (dist[i] > max) {
	  maxIndex = i;
	  max = dist[i];
	}
      }
      if (max > 0) {
	return maxIndex;
      } else {
	return Instance.missingValue();
      }
    case Attribute.NUMERIC:
      return dist[0];
    default:
      return Instance.missingValue();
    }
  }
}
