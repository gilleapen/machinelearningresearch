/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.core;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author Administrator
 */
public class FHX {
    
    private static void CreateReducedData(String trainFileName,int[] selectedAttributeSet)
            throws Exception
    {
         DataSource source = new DataSource(trainFileName);
         Instances train = source.getDataSet();
         Instances reduceData;
         Remove attributeFilter = new Remove();
         attributeFilter.setAttributeIndicesArray(selectedAttributeSet);
         attributeFilter.setInvertSelection(true);
         attributeFilter.setInputFormat(train);
         reduceData = Filter.useFilter(train, attributeFilter);
         String reducedFileName = trainFileName.substring(0,trainFileName.indexOf(".arff"));
         for(int i = 0; i < selectedAttributeSet.length -1; i++)
         {
             reducedFileName = reducedFileName.concat("_");
             reducedFileName = reducedFileName.concat(String.valueOf(selectedAttributeSet[i] + 1));
         }
         reducedFileName =reducedFileName.concat(".arff");
         MySave.SaveInstances(reduceData, reducedFileName);
    }
    private static void CreatReducedData(String trainFileName,String selectedIndexsFileName)
            throws Exception
    {
         DataSource source = new DataSource(trainFileName);
         Instances train = source.getDataSet();
         int classIndex = (train.classIndex() == -1 ? (train.numAttributes() -1) : train.classIndex());

         BufferedReader br = new BufferedReader(new FileReader(selectedIndexsFileName));
         String lineContent;
         String[] selectedIndexsStr;
         int[] selectedIndexs;
         while( (lineContent = br.readLine()) != null)
         {
              selectedIndexsStr = lineContent.split(",");
              selectedIndexs = new int[selectedIndexsStr.length + 1];
              for(int i = 0; i < selectedIndexsStr.length; i++)
              {
                   selectedIndexs[i] = Integer.valueOf(selectedIndexsStr[i]) -1 ;
              }
              selectedIndexs[selectedIndexsStr.length] = classIndex;
              CreateReducedData(trainFileName, selectedIndexs);
         }
    }

    public static void main(String[] args)
    {
        String[] options = (String[])args.clone();
        String trainFileName;
        String selectedIndexsFileName;
        try
        {
         trainFileName = Utils.getOption('T', options);
         selectedIndexsFileName = Utils.getOption('S', options);
         CreatReducedData(trainFileName,selectedIndexsFileName);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}
