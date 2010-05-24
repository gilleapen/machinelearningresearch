/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 *
 * @author Administrator
 */
public class MySave {

public static void  SaveInstances(Instances ti,String savefileName )
{
    if(savefileName.endsWith(".arff") == false)
    { savefileName = savefileName.concat(".arff");}
    BufferedWriter writer;
    try{
    writer = new BufferedWriter(new FileWriter(savefileName));
    writer.write(ti.toString());
    writer.flush();
    writer.close();
    }
    catch(Exception e)
    { e.printStackTrace(); }
}
public static void SaveStringBuffer(String buf,String savefileName)
{
    PrintWriter out;
    try{
    out= new PrintWriter(new BufferedWriter(new FileWriter(savefileName)));
    out.write(buf,0,buf.length());
    out.close();
    }
    catch(Exception e)
    { e.printStackTrace();}
}
}
