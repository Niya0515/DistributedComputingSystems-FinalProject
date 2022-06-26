package cn.edu.ecnu

import scala.collection.mutable.ArrayBuilder
import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import breeze.numerics.sqrt._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM}
import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.ml.common.{LabeledVector, Parameter, ParameterMap}
import org.apache.flink.api.scala._
import org.apache.flink.ml.math._
import org.apache.flink.ml.pipeline.{TransformOperation, FitOperation,Transformer}
import org.apache.flink.ml.preprocessing.InfoSelector.{Selected, NFeatures, Criterion, NI, NF, Dense}
import org.apache.flink.util.Collector
import org.apache.flink.api.scala.utils.`package`.DataSetUtils
import scala.collection.mutable.ArrayBuffer
import org.apache.flink.ml.common.FlinkMLTools
import org.apache.flink.configuration.Configuration
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.functions.RichFilterFunction
import org.apache.flink.api.common.functions.RichJoinFunction

case class F(feat: Int, crit: Double) 
case class ColumnarData(dense: DataSet[((Int, Int), Array[Byte])], 
    sparse: DataSet[(Int, BDV[Byte])],
    isDense: Boolean)

class InfoSelector extends Transformer[InfoSelector] {

  var selectedFeatures: Option[Array[Int]] = None
  private[preprocessing] var nfeatures: Int = 10
  private[preprocessing] var criterion: String = "mrmr" 

  def setSelected(s: Array[Int]): InfoSelector = {
    require(InfoSelector.isSorted(s))
    parameters.add(Selected, s)
    this
  }
  
  def setNFeatures(s: Int): InfoSelector = {
    require(s > 0)
    parameters.add(NFeatures, s)
    this
  }
  
  def setNF(s: Int): InfoSelector = {
    require(s > 0)
    parameters.add(NF, s)
    this
  }

  def setNI(s: Int): InfoSelector = {
    require(s > 0)
    parameters.add(NI, s)
    this
  }
  
  def setDense(s: Boolean): InfoSelector = {
    parameters.add(Dense, s)
    this
  }
  
  def setCriterion(s: String): InfoSelector = {
    parameters.add(Criterion, s)
    this
  }
}

object InfoSelector {

  // ====================================== Parameters =============================================

  case object Selected extends Parameter[Array[Int]] {
    override val defaultValue: Option[Array[Int]] = Some(Array(0))
  }
  
  case object NFeatures extends Parameter[Int] {
    val defaultValue = Some(10)
  }
  
  case object NF extends Parameter[Int] {
    val defaultValue = Some(631)
  }
  
  case object NI extends Parameter[Int] {
    val defaultValue = Some(631)
  }
  
  case object Dense extends Parameter[Boolean] {
    val defaultValue = Some(true)
  } 
  
  case object Criterion extends Parameter[String] {
    val defaultValue = Some("mrmr")
  }
  

  // ==================================== Factory methods ==========================================

  def apply(): InfoSelector = {
    new InfoSelector()
  }

  // ====================================== Operations =============================================

  implicit val fitLabeledVectorInfoSelector = {
    new FitOperation[InfoSelector, LabeledVector] {
      override def fit(
          instance: InfoSelector, 
          fitParameters: ParameterMap, 
          input: DataSet[LabeledVector]): Unit = {
        
        val map = instance.parameters ++ fitParameters

        // retrieve parameters of the algorithm
        val nselect = map(NFeatures)
        val nf = map(NF)
        val ni = map(NI)
        val dense = map(Dense)
        
        val critFactory = new InfoThCriterionFactory(map(Criterion))
        
        val selected = trainOn(input, critFactory, nselect, nf, ni, dense)
        require(isSorted(selected))
        instance.selectedFeatures = Some(selected)
      }
    }
  }
  
  private[InfoSelector] def isSorted(array: Array[Int]): Boolean = {
    var i = 1
    val len = array.length
    while (i < len) {
      if (array(i) < array(i-1)) return false
      i += 1
    }
    true
  }

 abstract class InfoSelectorTransformOperation extends TransformOperation[
        InfoSelector,
        Array[Int],
        LabeledVector,
        LabeledVector] {

    var selected: Array[Int] = _

   override def getModel(
      instance: InfoSelector, 
      transformParameters: ParameterMap): DataSet[Array[Int]] = {
      
      val env = ExecutionEnvironment.getExecutionEnvironment
      selected = transformParameters(Selected)
      
      val result = instance.selectedFeatures match {
        case Some(s) => env.fromElements((0, s))
        case None =>
          throw new RuntimeException("The InfoSelector has not been fitted to the data. ")
      }
      result.map(_._2)
    }

    def select(
        lp: LabeledVector, 
        filterIndices: Array[Int]) 
      : LabeledVector = {
       lp.vector match {
        case SparseVector(size, indices, values) =>
          val newSize = filterIndices.length
          val newValues = new ArrayBuilder.ofDouble
          val newIndices = new ArrayBuilder.ofInt
          var i = 0
          var j = 0
          var indicesIdx = 0
          var filterIndicesIdx = 0
          while (i < indices.length && j < filterIndices.length) {
            indicesIdx = indices(i)
            filterIndicesIdx = filterIndices(j)
            if (indicesIdx == filterIndicesIdx) {
              newIndices += j
              newValues += values(i)
              j += 1
              i += 1
            } else {
              if (indicesIdx > filterIndicesIdx) {
                j += 1
              } else {
                i += 1
              }
            }
          }
          // TODO: Sparse representation might be ineffective if (newSize ~= newValues.size)
          new LabeledVector(lp.label, new SparseVector(newSize, newIndices.result(), newValues.result()))
        case DenseVector(values) =>
          new LabeledVector(lp.label, new DenseVector(filterIndices.map(i => values(i))))
      }
    }
  }

  implicit val transformLabeledVector = {
    new InfoSelectorTransformOperation() {
      override def transform(
          vector: LabeledVector,
          model: Array[Int])
        : LabeledVector = {        
        select(vector, model)
      }
    }
  }

  private[preprocessing] def selectDenseFeatures(
      data: ColumnarData,
      criterionFactory: InfoThCriterionFactory, 
      nToSelect: Int,
      nInstances: Long,
      nFeatures: Int) = {
    
    val label = nFeatures - 1
    val it = InfoTheory.initializeDense(label, nInstances, nFeatures)
    val (mt, jt, rev, counter)  = it.initialize(data.dense)
    val initial = rev.map({ p => p._1 -> criterionFactory.getCriterion.init(p._2.toFloat) })
    
    // Print most relevant features
    val topByRelevance = initial.collect().sortBy(-_._2.score).take(nToSelect)
    val strRels = topByRelevance.map({case (f, c) => (f + 1) + "\t" + "%.4f" format c.score})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels)    
    
    val solution = if (criterionFactory.getCriterion.toString == "MIM") {
      rev.groupBy(1).sortGroup(1, Order.DESCENDING).first(nToSelect).collect().map({case (id, relv) => F(id, relv)}).reverse
    } else {      
      val func = new RichMapFunction[(Int, InfoThCriterion), (Int, InfoThCriterion)]() {
        var max: Array[(Int, Float)] = null
        
        override def open(config: Configuration): Unit = {
          // 3. Access the broadcasted DataSet as a Collection
          val aux = getRuntimeContext()
            .getBroadcastVariable[(Int, Float)]("max")
            .asScala.toArray
         max = aux
        }
        
        def map(c: (Int, InfoThCriterion)): (Int, InfoThCriterion) = { 
          if(c._1 == max(0)._1) c._1 -> c._2.setValid(false) else c
        }
      } 
      
      val func2 = new RichMapFunction[(Int, InfoThCriterion), (Int, InfoThCriterion)]() {
        var mivalues: Map[Int, (Float, Float)] = null
        
        override def open(config: Configuration): Unit = {
          // 3. Access the broadcasted DataSet as a Collection
          val aux = getRuntimeContext()
            .getBroadcastVariable[(Int, (Float, Float))]("mivalues")
            .asScala.toMap
         mivalues = aux
        }
        
        def map(c: (Int, InfoThCriterion)): (Int, InfoThCriterion) = { 
          if(c._2.valid) {
            val miv = mivalues.getOrElse(c._1, .0f -> .0f)
            c._1 -> c._2.update(miv._1, miv._2)
          } else {
            c
          }
        }
      }    
      val nIter = math.min(nToSelect, nFeatures) - 1
      val finalSelected = initial.iterate(nIter) { currentSelected =>   
          val omax: DataSet[(Int, Float)] = currentSelected
            .filter(_._2.valid)
            .map(c => c._1 -> c._2.score)
            .reduce( (c1, c2) => if(c1._2 > c2._2) c1 else c2)    
          val newInfo = it.getRedundancies(data.dense, omax, mt, jt, counter)
          
          currentSelected.map(func).withBroadcastSet(omax, "max").map(func2).withBroadcastSet(newInfo, "mivalues")
      }
      
      finalSelected.collect().filter(!_._2.valid).map({case (id, c) => F(id, c.score)}).toSeq
    }
    solution
  }
  
  /**
   * Performs a info-theory FS process.
   * 
   * @param data Columnar data (last element is the class attribute).
   * @param nInstances Number of samples.
   * @param nFeatures Number of features.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[preprocessing] def selectSparseFeatures(
      data: ColumnarData,
      criterionFactory: InfoThCriterionFactory, 
      nToSelect: Int,
      nInstances: Long,
      nFeatures: Int) = {
    
     throw new UnsupportedOperationException("selectSparseFeatures method is not implemented yet.")
  }

  /**
   * Process in charge of transforming data in a columnar format and launching the FS process.
   * 
   * @param data DataSet of LabeledPoint.
   * @return A feature selection model which contains a subset of selected features.
   * 
   */
  def trainOn(data: DataSet[LabeledVector], 
      criterionFactory: InfoThCriterionFactory, 
      nToSelect: Int,
      nFeatures: Int,
      nInstances: Int,
      dense: Boolean): Array[Int] = {
      
    // Feature vector must be composed of bytes, not the class
    println("Initializing feature selector...")
    val requireByteValues = (v: Vector) => {        
      val values = v match {
        case sv: SparseVector =>
          sv.data
        case dv: DenseVector =>
          dv.data
      }
      val condition = (value: Double) => value <= Byte.MaxValue && 
        value >= Byte.MinValue && value % 1 == 0.0
      if (!values.forall(condition(_))) {
        throw new IllegalArgumentException(s"Info-Theoretic Framework requires positive values in range [0, 255]")
      }           
    }
    
    // Get basic info    
    require(nToSelect < nFeatures)  
    
    // Start the transformation to the columnar format
    val colData = if(dense) {
      println("Performing the columnar transformation...")
      // Transform data into a columnar format by transposing the local matrix in each partition
      val denseIndexing = new RichMapPartitionFunction[LabeledVector, ((Int, Int), Array[Byte])]() {        
          def mapPartition(it: java.lang.Iterable[LabeledVector], out: Collector[((Int, Int), Array[Byte])]): Unit = {
            val index = getRuntimeContext().getIndexOfThisSubtask() // Partition index
            val mat = for (i <- 0 until nFeatures) yield new scala.collection.mutable.ListBuffer[Byte]
            for(reg <- it.asScala) {
              for (i <- 0 until (nFeatures - 1)) mat(i) += reg.vector(i).toByte
              mat(nFeatures - 1) += reg.label.toByte
            }
            for(i <- 0 until nFeatures) out.collect((i, index) -> mat(i).toArray) // numPartitions
          }
        }
      val denseData = data.mapPartition(denseIndexing).partitionByRange(0)
      val colpdata = FlinkMLTools.persist(denseData, "hdfs://bigdata:8020/tmp/dense-flink-columnar")      
      ColumnarData(colpdata, null, true)    
      
    } else {   
        
      // Transform data into a columnar format by transposing the local matrix in each partition
      val sparseIndexing = new RichMapPartitionFunction[(Long, LabeledVector), (Int, (Long, Byte))]() {        
        def mapPartition(it: java.lang.Iterable[(Long, LabeledVector)], out: Collector[(Int, (Long, Byte))]): Unit = {
          for((iinst, lp) <- it.asScala) {
            requireByteValues(lp.vector)
            val sv = lp.vector.asInstanceOf[SparseVector]
            val output = (nFeatures - 1) -> (iinst, lp.label.toByte)
            out.collect(output)
            for(i <- 0 until sv.indices.length) 
                out.collect(sv.indices(i) -> (iinst, sv.data(i).toByte))
          }
        }
      }
      val sparseData = data.zipWithIndex.mapPartition(sparseIndexing)
      
      // Transform sparse data into a columnar format 
      // by grouping all values for the same feature in a single vector
      val columnarData = sparseData.groupBy(0).reduceGroup{ it =>
          //val a = it.toArray
          var k: Int = -1
          val init = Array.fill[Byte](nInstances.toInt)(0)
          val result: BDV[Byte] = new BDV(init)
          //if(a.size >= nInstances) {
          for((fk, (iind, v)) <- it){
            k = fk
            result(iind.toInt) = v
          }
          k -> result // Feature index -> array of cells
        }.partitionByHash(0)
      val colpdata = FlinkMLTools.persist(columnarData, "hdfs://bigdata:8020/tmp/sparse-flink-columnar")
      ColumnarData(null, colpdata, false)
    }
    
    // Start the main algorithm
    println("Starting to select features...")
    val selected = if(dense) {
      selectDenseFeatures(colData, criterionFactory, nToSelect, 
        nInstances, nFeatures)
    } else {
      selectSparseFeatures(colData, criterionFactory, nToSelect, 
        nInstances, nFeatures)
    }
  
    // Print best features according to the mRMR measure
    val out = selected.sortBy{ case F(feat, rel) => -rel}.map{case F(feat, rel) => 
        (feat + 1) + "\t" + "%.4f".format(rel)
      }.mkString("\n")
    println("\n*** Selected features ***\nFeature\tScore\n" + out)
    // Features must be sorted
    selected.map{case F(feat, rel) => feat}.sorted.toArray
  }
}
