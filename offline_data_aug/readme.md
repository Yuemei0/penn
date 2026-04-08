目的：增强模型的跨域泛化能力
将mdb/ptdb的数据集做数据增强（音高搬移，频谱包络扭曲，环境退化等 ）。
简单起见，采用offline的生成方式，将增强后的数据另保存一个地址。
注意：
    源数据和增强的数据必须在同一个集中。
    对train和valid做增强。




Onenote笔记链接：
https://onedrive.live.com/view.aspx?resid=70B592AD55407D2A%21715&id=documents&wd=target%28%E6%96%B0%E5%88%86%E5%8C%BA%201.one%7CBC96ED86-D5B5-438C-8B0F-A5BFFC267E83%2F20260408%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%E8%B7%AF%E7%BA%BF%7C753EC9A0-CB1D-4D9D-884B-A6B808DB9BFF%2F%29&wdpartid={89F7B9E6-E9D1-0852-3FF7-8268E6B2A0A0}{1}&wdsectionfileid=70B592AD55407D2A!se95ae0c23f914a22b021af21ce25eec8&end
onenote:https://d.docs.live.net/70B592AD55407D2A/文档/CASIA/新分区%201.one#20260408数据增强路线&section-id={BC96ED86-D5B5-438C-8B0F-A5BFFC267E83}&page-id={753EC9A0-CB1D-4D9D-884B-A6B808DB9BFF}&end