package eggman89


class hashmap  extends java.io.Serializable
{
  var obj:Map[String,Int] = Map()
  var id = -1
  def add(value:String): Int ={

    if (obj.contains(value) == true)
    {
      if (value == "?")
      {
        return 0;
      }
      obj(value)
    }

    else
    {      id = id + 1

      obj = obj +(value->id)
      id
    }
  }

  def findval(value : Int) : String = {
    val default = ("-1",0)
    obj.find(_._2==value).getOrElse(default)._1
  }
}
