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
 * RankedInstance.java
 * Copyright (C) 2005 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import weka.core.Instance;
import weka.core.InstanceComparator;

import java.io.Serializable;

/**
 * This class is used for storing the distributions generated by the different
 * trees per instance. It only works with binary classes!
 * 
 *
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version   $Revision$
 */
public class RankedInstance 
  implements Comparable, Serializable {
    
  /** for serialization */
  private static final long serialVersionUID = 6458682847912963957L;

  /** the instance we store the distributions for */
  private Instance m_Instance = null;

  /** the accumulated distributions */
  private double[] m_Distribution = null;

  /** the number of distributions added so far */
  private int m_NumberOfDistributions;

  /** the comparator for determine whether an instance is the same as the
   *  stored one (class attribute is excluded from comparison) */
  private static InstanceComparator m_Comparator = null;

  /**
   * initializes with 0 distributions so far
   *
   * @param inst        the instance to store the distributions for
   */
  public RankedInstance(Instance inst) {
    super();

    m_Instance              = inst;
    m_NumberOfDistributions = 0;
    m_Distribution          = new double[2];
    for (int i = 0; i < 2; i++)
      m_Distribution[i] = 0;

    if (m_Comparator == null)
      m_Comparator = new InstanceComparator(false);
  }

  /**
   * initializes with the given distribution (e.g., class distribution)
   * 
   * @param inst        the instance to store the distributions for
   * @param dist        the initial distribution, e.g., class distribution
   */
  public RankedInstance(Instance inst, double[] dist) {
    this(inst);

    addDistribution(dist);
  }

  /**
   * returns the instance we collect the distributions for
   * 
   * @return		the instance
   */
  public Instance getInstance() {
    return m_Instance;
  }

  /**
   * returns the normalized distribution
   * 
   * @return		the normalized distribution
   */
  public double[] getDistribution() {
    double[]      dist;
    int           i;

    dist = new double[m_Distribution.length];

    if (m_NumberOfDistributions > 0) {
      for (i = 0; i < m_Distribution.length; i++)
        dist[i] = m_Distribution[i] / m_NumberOfDistributions;
    }
    else {
      for (i = 0; i < m_Distribution.length; i++)
        dist[i] = m_Distribution[i];
    }

    return dist;
  }

  /**
   * adds the distribution
   * 
   * @param dist	the disttribution to add
   */
  public void addDistribution(double[] dist) {
    int         i;
    
    m_NumberOfDistributions++;
    for (i = 0; i < m_Distribution.length; i++)
      m_Distribution[i] += dist[i];
  }

  /**
   * Compares this object with the specified object for order. Can compare
   * either another CollectiveRankedInstance or just an Instance.
   * 
   * @param o		the object to compare with
   * @return		the comparison result
   */
  public int compareTo(Object o) {
    int       result;
    
    if (o instanceof Instance)
      result = m_Comparator.compare(getInstance(), (Instance) o);
    else if (o instanceof RankedInstance)
      result = m_Comparator.compare(getInstance(), ((RankedInstance) o).getInstance());
    else
      result = 0;

    return result;
  }

  /**
   * returns a string representation of this object
   * 
   * @return 		a string representation
   */
  public String toString() {
    String          result;

    result = getInstance().toString() + "\n";
    result +=   "  class A = " + getDistribution()[0]
              + ", class B = " + getDistribution()[1];

    return result;
  }
}
