package cn.edu.ecnu

import org.apache.hadoop.fs.Path

import org.apache.spark.annotation.{ Since, Experimental }
import org.apache.spark.ml._
import org.apache.spark.ml.attribute.{ AttributeGroup, _ }
import org.apache.spark.ml.linalg.{ Vector, VectorUDT }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.{ Vectors => OldVectors }
import org.apache.spark.mllib.regression.{ LabeledPoint => OldLabeledPoint }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ DoubleType, StructField, StructType }

import org.apache.spark.mllib.feature.InfoThCriterionFactory

private[feature] trait InfoThSelectorParams extends Params
    with HasFeaturesCol with HasOutputCol with HasLabelCol {

  val MIM = "mim"
  val MIFS = "mifs"
  val JMI = "jmi"
  val MRMR = "mrmr"
  val ICAP = "icap"
  val CMIM = "cmim"
  val IF = "if"

  final val selectCriterion = new Param[String](this, "selectCriterion",
    "Information Theoretic criterion used to rank the features. The criterion to be chosen are: (mim, mifs, jmi, mrmr, icap, cmim, if).")
  setDefault(selectCriterion -> "mrmr")

  final val numTopFeatures = new IntParam(this, "numTopFeatures",
    "Number of features that selector will select, ordered by statistics value descending. If the" +
      " number of features is < numTopFeatures, then this will select all features.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures -> 25)

  final val nPartitions = new IntParam(this, "nPartitions",
    "Number of partitions to use after the data matrix is transformed to a columnar format.",
    ParamValidators.gtEq(0))
  setDefault(nPartitions -> 0)

  def getSelectCriterion: String = $(selectCriterion)
  def getNumTopFeatures: Int = $(numTopFeatures)
  def getNPartitions: Int = $(nPartitions)
}

@Experimental
final class InfoThSelector @Since("1.6.0") (@Since("1.6.0") override val uid: String)
    extends Estimator[InfoThSelectorModel] with InfoThSelectorParams with DefaultParamsWritable {

  @Since("1.6.0")
  def this() = this(Identifiable.randomUID("InfoThSelector"))

  @Since("1.6.0")
  def setSelectCriterion(value: String): this.type = set(selectCriterion, value)

  @Since("1.6.0")
  def setNumTopFeatures(value: Int): this.type = set(numTopFeatures, value)

  @Since("1.6.0")
  def setNPartitions(value: Int): this.type = set(nPartitions, value)

  @Since("1.6.0")
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  @Since("1.6.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  @Since("1.6.0")
  def setLabelCol(value: String): this.type = set(labelCol, value)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): InfoThSelectorModel = {
    transformSchema(dataset.schema, logging = true)
    val input: RDD[OldLabeledPoint] =
      dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          OldLabeledPoint(label, OldVectors.fromML(features))
      }.cache()

    val InfoThSelector = new feature.InfoThSelector(
      new InfoThCriterionFactory($(selectCriterion)),
      $(numTopFeatures),
      $(nPartitions)
    ).fit(input)
    copyValues(new InfoThSelectorModel(uid, InfoThSelector).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }

  override def copy(extra: ParamMap): InfoThSelector = defaultCopy(extra)
}

@Since("1.6.0")
object InfoThSelector extends DefaultParamsReadable[InfoThSelector] {

  @Since("1.6.0")
  override def load(path: String): InfoThSelector = super.load(path)
}

@Experimental
final class InfoThSelectorModel private[ml] (
  @Since("1.6.0") override val uid: String,
  private val InfoThSelector: feature.InfoThSelectorModel
)
    extends Model[InfoThSelectorModel] with InfoThSelectorParams with MLWritable {

  import InfoThSelectorModel._

  val selectedFeatures: Array[Int] = InfoThSelector.selectedFeatures
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformedSchema = transformSchema(dataset.schema, logging = true)
    val newField = transformedSchema.last

    // Make the transformer natively in ml framework to avoid extra conversion.
    val transformer: Vector => Vector = v => InfoThSelector.transform(OldVectors.fromML(v)).asML

    val selector = udf(transformer)

    dataset.withColumn($(outputCol), selector(col($(featuresCol))), newField.metadata)
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    val newField = prepOutputField(schema)
    val outputFields = schema.fields :+ newField
    StructType(outputFields)
  }

  /**
   * Prepare the output column field, including per-feature metadata.
   */
  private def prepOutputField(schema: StructType): StructField = {
    val selector = InfoThSelector.selectedFeatures.toSet
    val origAttrGroup = AttributeGroup.fromStructField(schema($(featuresCol)))
    val featureAttributes: Array[Attribute] = if (origAttrGroup.attributes.nonEmpty) {
      origAttrGroup.attributes.get.zipWithIndex.filter(x => selector.contains(x._2)).map(_._1)
    } else {
      Array.fill[Attribute](selector.size)(NominalAttribute.defaultAttr)
    }
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes)
    newAttributeGroup.toStructField()
  }

  override def copy(extra: ParamMap): InfoThSelectorModel = {
    val copied = new InfoThSelectorModel(uid, InfoThSelector)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new InfoThSelectorModelWriter(this)
}

@Since("1.6.0")
object InfoThSelectorModel extends MLReadable[InfoThSelectorModel] {

  private[InfoThSelectorModel] class InfoThSelectorModelWriter(instance: InfoThSelectorModel) extends MLWriter {

    private case class Data(selectedFeatures: Seq[Int])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.selectedFeatures.toSeq)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class InfoThSelectorModelReader extends MLReader[InfoThSelectorModel] {

    private val className = classOf[InfoThSelectorModel].getName

    override def load(path: String): InfoThSelectorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("selectedFeatures").head()
      val selectedFeatures = data.getAs[Seq[Int]](0).toArray
      val oldModel = new feature.InfoThSelectorModel(selectedFeatures)
      val model = new InfoThSelectorModel(metadata.uid, oldModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[InfoThSelectorModel] = new InfoThSelectorModelReader

  @Since("1.6.0")
  override def load(path: String): InfoThSelectorModel = super.load(path)
}