/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 * /*
<!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 *    title = {Semi-supervised Feature Selection via Spectral Analysis.pdf}
 *    本算法只能处理两类问题
 * </pre>
 <!-- technical-bibtex-end -->
*/
package weka.attributeSelection;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.DistanceFunction; //距离函数
import weka.core.matrix.Matrix;//矩阵
import weka.core.EuclideanDistance;//欧拉距离

import java.util.Enumeration;
import java.util.Vector;
import weka.core.ContingencyTables;

/**
 *
 * @author fanghongxia
 * @Email:fanghongxia2008@yahoo.cn
 * //模板来自：ReliefFAttributeEval.java
 */
public class SemiSupSSelectEval extends ASEvaluation
  implements SemiASEvaluation,
             AttributeEvaluator,
             OptionHandler,
             TechnicalInformationHandler {

  /** for serialization 序列化所需*/
  static final long serialVersionUID = -8422186665795839379L;

  /** The training instances 训练集 */
  private Instances m_trainInstances;

  /** 标记样本则为true，否则false */
  private boolean[] m_IsLabeled;

  /** 标记样本数目 */
  private int m_numLabeled;

  /** The class index 类别属性索引 */
  private int m_classIndex;

  /** The number of attributes 特征数目*/
  private int m_numAttribs;

  /** The number of instances 样本数目*/
  private int m_numInstances;

  /** The number of classes */
  private int m_numClasses;//必须是两类问题，才可处理

  /** Holds the weights that sSelect assigns to attributes */
  private double[] m_weights;

//--------------------------------------------------------------------------
  /** The arg of RBF kernel function ( > 0.0,default value is 1.0)*/
  private double m_sigma;
  /** The arg of nearest number ( >= 1 ,dedault value is 10)*/
  private int m_Knn;
  /** The arg of sSelect Algorithm (between 0.0 and 1.0,default value is 0.5)*/
  private double m_lambda;

  /**
  /** The DistanceFunction (= new EuclideanDistance())*/
  private DistanceFunction m_DistanceFunction ; 
//--------------------------------------------------------------------------
  private double[][] m_w;
  private Matrix m_MatrixW;
  private Matrix m_MatrixD;
  private Matrix m_MatrixL;
  private double[] m_d;
  private double m_VolV;
  private Vector m_Yvalues;

 //-------------------------------------------------------------------------

  /**
   * Constructor
   */
  public SemiSupSSelectEval () {
    resetOptions();
    m_DistanceFunction = new EuclideanDistance();//欧式距离
  }

  /**
   * Returns a string describing this attribute evaluator
   * @return a description of the evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "SemiSupSSelectEval :\n\nEvaluates the worth of an attribute using "
      +"sSelect Algorithm which is detailed introduced by Semi-supervised "
      +"Feature Selection via Spectral Analysis.pdf.\n\n"
      + "For more information see:\n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Return a description of the sSelect attribute evaluator.
   *
   * @return a description of the evaluator as a String.
   */
  @Override
  public String toString () {
    StringBuffer text = new StringBuffer();

    if (m_trainInstances == null) {
      text.append("SSelect Algorithm feature evaluator has not been built yet\n");
    }
    else {
      text.append("\tSSelect Ranking Filter");
      text.append("\n\tInstances sampled: ");
      text.append("\tRBF kernel function arg sigma: " + m_sigma + "\n");
      text.append("\tNumber of nearest neighbours (k): " + m_Knn + "\n");
      text.append("\tWeight of labeled against unlabeled arg lambda: " + m_lambda + "\n");
    }
    return  text.toString();
  }

  /**
   * Returns the capabilities of this evaluator.
   *
   * @return            the capabilities of this evaluator
   * @see               Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   *
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation    result;
    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Zhao, Z.and Liu, H.");
    result.setValue(Field.TITLE, "Semi-supervised Feature Selection via Spectral Analysis");
    result.setValue(Field.YEAR, "2007");
    result.setValue(Field.PAGES, "1151–1158");
    result.setValue(Field.PUBLISHER, "Citeseer");
    return result;
  }

  /**
   * Returns an enumeration describing the available options.
   * @return an enumeration of all the available options.
   **/
  public Enumeration listOptions () {
    Vector newVector = new Vector(3);

    newVector.
      addElement(new Option("\tSpecify sigma value (used in an exp\n"
                            + "\tfunction to control how quickly\n"
                            + "\tweights for more distant instances\n"
                            + "\tdecrease.(Default = 1)\n"
                            + "\t", "S", 1, "-S <sigma>"));
    newVector.
      addElement(new Option("\tNumber of nearest neighbours (k) used\n"
                            + "\tto estimate attribute relevances\n"
                            + "\t(Default = 10).", "K", 1
                            , "-K <Knn>"));
    newVector
      .addElement(new Option("\tSpecify the weights between labeled and\n"
                             + "\tunlabeled  sample when estimating attributes.\n"
                             +"" , "L", 1, "-L <lambda>"));
    
    return  newVector.elements();
  }

  /**
  *
  * @param options
  * @throws Exception
  */
  public void setOptions (String[] options)throws Exception {
    String optionString;
    resetOptions();

    optionString = Utils.getOption('S', options); //m_sigma
    setSigma(Double.parseDouble(optionString));

    optionString = Utils.getOption('K', options); //m_Knn
    setNumNeighbours(Integer.parseInt(optionString));

    optionString = Utils.getOption('L', options); //m_lambda
    setLambda(Double.parseDouble(optionString));
  }

  /**
   * Gets the current settings of sSelect Algorithm.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions () {

    String[] options = new String[6];
    int current = 0;

    options[current++] = "-S";
    options[current++] = "" + getSigma(); //m_sigma

    options[current++] = "-K";
    options[current++] = "" + getNumNeighbours(); //m_Knn

    options[current++] = "-L";
    options[current++] = "" + getLambda(); //m_lambda

    return  options;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String sigmaTipText() {
    return "Set arg of RBF kernel funcion,default value is 1.0 ";
  }

  /**
   * Sets the sigma value.
   *
   * @param s the value of sigma (> 0)
   * @throws Exception if s is not positive
   */
  public void setSigma (double s)throws Exception {
    if (s <= 0.0) {
      throw  new Exception("value of sigma must be > 0.0!");
    }

    m_sigma = s;
  }

  /**
   * Get the value of sigma.
   *
   * @return the sigma value.
   */
  public double getSigma () {
    return  m_sigma;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numNeighboursTipText() {
    return "Number of nearest neighbours for attribute estimation.";
  }

  /**
   * Set the number of nearest neighbours
   *
   * @param n the number of nearest neighbours.
   */
  public void setNumNeighbours (int n) throws Exception{
    if (n < 1) {
      throw  new Exception("value of Neighbours must be >= 1!");
    }
    m_Knn = n;

  }

  /**
   * Get the number of nearest neighbours
   *
   * @return the number of nearest neighbours
   */
  public int getNumNeighbours () {
    return  m_Knn;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String lambdaTipText() {
    return "Set arg of sSelect Algorithm,default value is 0.5 ";
  }

  /**
   * Sets the lambda value.
   *
   * @param s the value of lambda (>=0.0 && <= 1.0)
   * @throws Exception if s is not between 0.0 and 1.0
   */
  public void setLambda (double s)throws Exception {
    if (s < 0.0 || s > 1.0) {
      throw  new Exception("value of sigma must between 0.0 and 1.0!");
    }

    m_lambda = s;
  }

  /**
   * Get the value of lambda.
   *
   * @return the lamda value.
   */
  public double getLambda () {
    return  m_lambda;
  }

  /**
   * Initializes a ReliefF attribute evaluator.
   *
   * @param data set of instances serving as training data
   * @throws Exception if the evaluator has not been
   * generated successfully
   */
  public void buildEvaluator (Instances data)throws Exception {

    // can evaluator handle data?
    getCapabilities().testWithFail(data);

    m_trainInstances = data; //训练数据
    m_classIndex = m_trainInstances.classIndex(); //类别索引
    m_numAttribs = m_trainInstances.numAttributes(); //特征数目
    m_numInstances = m_trainInstances.numInstances(); //样本数目

    Vector a  = new Vector(2);
    double[] classArray = m_trainInstances.attributeToDoubleArray(m_classIndex);
    for(int i = 0; i < m_numInstances; i++)
    {
        if(m_trainInstances.instance(i).classIsMissing() == false)
        {
            if( a.contains((Double)classArray[i]) == false )
            { a.add((Double)classArray[i]);}
        }
    }
    m_Yvalues = a;
    if(m_trainInstances.attribute(m_classIndex).isNumeric())
    {    m_numClasses = a.size();  }
    else {m_numClasses = m_trainInstances.attribute(m_classIndex).numValues();}
    if(m_numClasses != 2) {  throw new Exception("本方法只可以处理两类问题！"); }
    m_DistanceFunction =new EuclideanDistance();
    m_DistanceFunction.setInstances(m_trainInstances);
    compute_w(); //double[][] m_w
    computeMatrixW(); //Matrix m_MatrixW
    compute_d(); //double[] m_d
    computeMatrixD(); //Matrix m_MatrixD
    computeMatrixL(); //Matrix m_MatrixL
    computeVolV(); //double m_VolV
    
    m_IsLabeled = new boolean[m_numInstances];
    m_numLabeled = 0;
    for(int i = 0; i < m_numInstances; i++)
    {
        if(m_trainInstances.instance(i).classIsMissing() == false)
         {  m_IsLabeled[i] = true; m_numLabeled ++;}
        else  {  m_IsLabeled[i] = false; }
    }

    // the final attribute weights
    m_weights = new double[m_numAttribs];
    double[] f,g;
    double[] g_yiba, y = compute_y();
    double w_unsup,w_sup;
    for(int i= 0;i < m_numAttribs;i++)
    {
        if( i != m_classIndex )
        {
            f = m_trainInstances.attributeToDoubleArray(i);// 特征i列向量
            g = compute_phi(f); //<g,d> = 0
            w_unsup = compute_Punsup(g); //无监督部分的权重

            g_yiba = compute_g_yiba(g);
            w_sup = compute_Psup(g_yiba, y); //监督部分的权重

            m_weights[i] = m_lambda * w_unsup + (1.0 - m_lambda) * w_sup;
        }
    }
  }

  /**
   * Evaluates an individual attribute using ReliefF's instance based approach.
   * The actual work is done by buildEvaluator which evaluates all features.
   *
   * @param attribute the index of the attribute to be evaluated
   * @throws Exception if the attribute could not be evaluated
   */
  public double evaluateAttribute (int attribute)throws Exception {
    return  m_weights[attribute];
  }

  /**
   * Reset options to their default values
   */
  protected void resetOptions () {
    m_trainInstances = null;
    m_sigma = 1.0;
    m_Knn = 10;
    m_lambda = 0.5;
  }

  /**
   * Returns the revision string.
   *
   * @return		the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.23 $");
  }

  /**
   *
   * @param labeledData
   * @param unlabledData
   * @throws Exception
   */
  public void buildEvaluator(Instances labeledData, Instances unlabledData) throws Exception {
        
      // can evaluator handle data?
    for(int i =0;i < labeledData.numInstances();i++)
    {
        if(labeledData.instance(i).classIsMissing() == true)
        throw new Exception("Some Labeled Instances is classmissing!");
    }
    m_trainInstances = new Instances(labeledData); //训练数据
    for(int i = 0; i < unlabledData.numInstances(); i++)
    {   m_trainInstances.add(unlabledData.instance(i)); }
    
    getCapabilities().testWithFail(m_trainInstances);
    m_classIndex = m_trainInstances.classIndex(); //类别索引
    m_numAttribs = m_trainInstances.numAttributes(); //特征数目
    m_numInstances = m_trainInstances.numInstances(); //样本数目
    
    Vector a  = new Vector(2);
    double[] classArray = m_trainInstances.attributeToDoubleArray(m_classIndex);
    for(int i = 0; i < m_numInstances; i++)
    {
        if(m_trainInstances.instance(i).classIsMissing() == false)
        {
            if( a.contains((Double)classArray[i]) == false )
            { a.add((Double)classArray[i]);}
        }
    }
    m_Yvalues = a;
    if(m_trainInstances.attribute(m_classIndex).isNumeric())
    {    m_numClasses = a.size();  }
    else {m_numClasses = m_trainInstances.attribute(m_classIndex).numValues();}
    if(m_numClasses != 2) {  throw new Exception("本方法只可以处理两类问题！"); }
    m_DistanceFunction =new EuclideanDistance();
    m_DistanceFunction.setInstances(m_trainInstances);
    
    compute_w(); //double[][] m_w
    computeMatrixW(); //Matrix m_MatrixW
    compute_d(); //double[] m_d
    computeMatrixD(); //Matrix m_MatrixD
    computeMatrixL(); //Matrix m_MatrixL
    computeVolV(); //double m_VolV
    
    m_IsLabeled = new boolean[m_numInstances];
    m_numLabeled = labeledData.numInstances();
    for(int i = 0; i < labeledData.numInstances(); i++)
    {  m_IsLabeled[i] = true; }
    for(int i = labeledData.numInstances(); i < m_numInstances; i++)
    {  m_IsLabeled[i] = false; }

    // the final attribute weights
    m_weights = new double[m_numAttribs];
    double[] f,g;
    double[] g_yiba, y = compute_y();
    double w_unsup,w_sup;
    for(int i= 0;i < m_numAttribs;i++)
    {
        if( i != m_classIndex )
        {
            f = m_trainInstances.attributeToDoubleArray(i);// 特征i列向量
            g = compute_phi(f); //<g,d> = 0
            w_unsup = compute_Punsup(g); //无监督部分的权重

            g_yiba = compute_g_yiba(g);
            w_sup = compute_Psup(g_yiba, y); //监督部分的权重

            m_weights[i] = m_lambda * w_unsup + (1.0 - m_lambda) * w_sup;
        }
    }
      throw new UnsupportedOperationException("Not supported yet.");
    }

  /**
   *
   * @return
   */
  private double[] compute_y()
  {
      double[] y = new double[m_numLabeled];
      int k = 0;
      for(int i = 0; i < m_numInstances;i++)
      {
          if(m_IsLabeled[i] == true )
          { y[k] = m_trainInstances.instance(i).classValue(); k++; }
      }
      return y;
  }

  /**
   *
   * @param g
   * @return
   */
  private double[] compute_g_yiba(double[] g)
  {
    double[] g_yiba = new double[m_numLabeled];
    int k = 0;
    for(int i = 0; i < m_numInstances; i++)
    {
        if(m_IsLabeled[i] == true)
        {
            if( g[i] >= 0.0){ g_yiba[k] = 1.0; }
            else { g_yiba[k] = -1.0; }
            k++;
        }
    }
    return g_yiba;
  }

  private double compute_Psup(double[] g,double[] y) throws Exception
  {
      if(g.length != y.length ) 
      throw new Exception("Not supported yet.");
      int length = g.length;
      double[] gvalue = new double[]{-1.0, 1.0};
      double[] yvalue = new double[2];
      yvalue[0] = Double.parseDouble(m_Yvalues.elementAt(0).toString());
      yvalue[1] = Double.parseDouble(m_Yvalues.elementAt(1).toString());

      double[] gcount = new double[] {0.0,0.0}; //符号化数字{-1.0，+1.0}两类
      double[] ycount = new double[] {0.0,0.0}; //两类问题
      double[][] gycount =new double[][] {{0.0,0.0},{0.0,0.0}}; //联合分布

      for(int i =0; i < length; i++)
      {
          if(Math.abs( g[i] - gvalue[0] ) <  Math.abs(g[i] - gvalue[1]))
          {
              gcount[0] += 1.0;
              if(Math.abs( y[i] - yvalue[0] ) <  Math.abs(y[i] - yvalue[1]))
              {   ycount[0] += 1.0; gycount[0][0] += 1.0;}
              else
              {   ycount[1] += 1.0; gycount[0][1] += 1.0;}
          }
          else
          {
              gcount[1] += 1.0;
              if(Math.abs( y[i] - yvalue[0] ) <  Math.abs(y[i] - yvalue[1]))
              {   ycount[0] += 1.0; gycount[1][0] += 1.0;}
              else
              {   ycount[1] += 1.0; gycount[1][1] += 1.0;}
          }
      }
      double H_g = ContingencyTables.entropy(gcount);
      double H_y = ContingencyTables.entropy(ycount);
      gcount[0] /= length; gcount[1] /= length;
      ycount[0] /= length; ycount[1] /= length;
      gycount[0][0] /= length;gycount[0][1] /= length;
      gycount[1][0] /= length;gycount[1][1] /= length;
      double NMI_GY =
               (gycount[0][0] * Math.log10(gycount[0][0]/(gcount[0]*ycount[0])))
              +(gycount[0][1] * Math.log10(gycount[0][1]/(gcount[0]*ycount[1])))
              +(gycount[1][0] * Math.log10(gycount[1][0]/(gcount[1]*ycount[0])))
              +(gycount[1][1] * Math.log10(gycount[1][1]/(gcount[1]*ycount[1])));
      NMI_GY /= Math.log(2);
      NMI_GY /= Math.max(H_g, H_y);
      return NMI_GY;
  }

  /**
   * parter of unsup weight
   */
  private double compute_Punsup(double[] g)
  {
      double s_numerator = 0.0;//分子
      double s_denominator = 0.0; //分母
      //compute s_numerator;
      for(int i = 0; i < m_numInstances; i++)
      {
          for(int j = 0; j < m_numInstances; j++)
          {  s_numerator += (g[i] -g[j]) * (g[i] -g[j]) * m_w[i][j];}
      }
      //compute s_denominator
      for(int i = 0;i <m_numInstances; i++)
      {  s_denominator += g[i] * g[i] * m_d[i];}

      s_denominator *= 2.0;

      return s_numerator/s_denominator;
  }

  /**
  /**
   *
   * @param f Attribute values
   */
  private double[] compute_phi(double[] f)
  {
      double[] g = new double[m_numInstances];
      double sum_fd = 0.0;
      for(int i = 0; i < m_numInstances; i++)
      {  sum_fd += f[i] * m_d[i];}
      sum_fd = sum_fd / m_VolV;
      for(int i = 0; i < m_numInstances; i++)
      {  g[i] = f[i] - sum_fd;  }

      return g;
  }

  /**
     * compute the m_VolV field
     */
  private void computeVolV()
  {
      double volV =0.0;
      for(int i =0; i < m_numInstances; i++)
      {   volV += m_d[i]; }
  }

  /**
     * compute the m_MatrixL field
     */
  private void computeMatrixL()
  {
       m_MatrixL = m_MatrixD.minus(m_MatrixW);
    }

  /**
     * compute the m_MatrixD field
     */
  private void computeMatrixD()
  {
        m_MatrixD = new Matrix(m_numInstances,m_numInstances,0.0);
        for(int i = 0; i < m_numInstances; i++)
        {
            m_MatrixD.set(i,i,m_d[i]);
        }
    }

  /**
     * compute the m_d field
     */
  private void compute_d()
  {
        m_d = new double[m_numInstances];
        for(int i = 0; i < m_numInstances; i++)
        {
            m_d[i] = 0.0;
            for(int j = 0; j < m_numInstances; j++)
            {
                m_d[i] += m_MatrixW.get(i, j);
            }
        }
    }

  /**
     *compute the m_MatrixW field
     */
  private void computeMatrixW()
  {
        int[][] maxkIndexs = new int[m_numInstances][m_Knn];
        int tempMin;
        for(int i = 0; i < m_numInstances; i++)
        {
            for(int j = 0; j < m_Knn; j++)
            { maxkIndexs[i][j] = j;}

            tempMin = 0;
            for(int k = 1; k < m_Knn; k++)
            {
                if( m_w[i][maxkIndexs[i][k]] < m_w[i][maxkIndexs[i][tempMin]] )
                { tempMin = k;}
            }

            for(int j = m_Knn;j < m_numInstances; j++)
            {
                if(m_w[i][j] > m_w[i][maxkIndexs[i][tempMin]])
                maxkIndexs[i][tempMin]  = j;

                tempMin = 0;
                for(int k = 1; k < m_Knn; k++)
                {
                    if( m_w[i][maxkIndexs[i][k]] < m_w[i][maxkIndexs[i][tempMin]] )
                    { tempMin = k;}
                }
            }
        }
        //
        m_MatrixW = new Matrix(m_numInstances,m_numInstances,0.0);
        for(int i = 0; i < m_numInstances; i++ )
        {
            for(int j = 0; j < m_Knn; j++)
            {
                m_MatrixW.set(i,maxkIndexs[i][j],m_w[i][maxkIndexs[i][j]]);
            }
        }
    }

  /**
     * compute the m_w field
     */
  private void compute_w()
  {
        m_w =new double[m_numInstances][m_numInstances];
        for(int i = 0; i < m_numInstances; i++)
        {
            for(int j = 0; j <= i; j++)
            {
                m_w[i][j] = m_w[j][i] = RBFKernelFunction(
                        m_trainInstances.instance(i),m_trainInstances.instance(j));
            }
        }
    }

  /**
     * compute the RBF kernel function in ordor to compute m_w
     * RBF核函数
     * @param a the first instance
     * @param b the second instance
     * @return the result of RBF kernel function
     */
  private double RBFKernelFunction(Instance a,Instance b)
  {
      double RBFResult;
      double d = m_DistanceFunction.distance(a, b);
      RBFResult = StrictMath.exp(-d / (2 * getSigma() * getSigma()));
      return RBFResult;
    }
}
