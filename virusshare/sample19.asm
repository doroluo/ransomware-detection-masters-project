
sample1:     file format pei-i386


Disassembly of section .text:

00401000 <.text>:
  401000:	83 ec 0c             	sub    $0xc,%esp
  401003:	53                   	push   %ebx
  401004:	55                   	push   %ebp
  401005:	56                   	push   %esi
  401006:	57                   	push   %edi
  401007:	68 40 00 00 f0       	push   $0xf0000040
  40100c:	33 ed                	xor    %ebp,%ebp
  40100e:	c7 44 24 1c 10 00 00 	movl   $0x10,0x1c(%esp)
  401015:	00 
  401016:	6a 01                	push   $0x1
  401018:	55                   	push   %ebp
  401019:	55                   	push   %ebp
  40101a:	8d 44 24 24          	lea    0x24(%esp),%eax
  40101e:	89 6c 24 24          	mov    %ebp,0x24(%esp)
  401022:	50                   	push   %eax
  401023:	8b fa                	mov    %edx,%edi
  401025:	89 6c 24 24          	mov    %ebp,0x24(%esp)
  401029:	8b d9                	mov    %ecx,%ebx
  40102b:	8b f5                	mov    %ebp,%esi
  40102d:	ff 15 00 60 40 00    	call   *0x406000
  401033:	85 c0                	test   %eax,%eax
  401035:	74 62                	je     0x401099
  401037:	8d 44 24 10          	lea    0x10(%esp),%eax
  40103b:	50                   	push   %eax
  40103c:	55                   	push   %ebp
  40103d:	55                   	push   %ebp
  40103e:	68 03 80 00 00       	push   $0x8003
  401043:	ff 74 24 24          	push   0x24(%esp)
  401047:	ff 15 10 60 40 00    	call   *0x406010
  40104d:	85 c0                	test   %eax,%eax
  40104f:	74 3d                	je     0x40108e
  401051:	55                   	push   %ebp
  401052:	ff 74 24 24          	push   0x24(%esp)
  401056:	57                   	push   %edi
  401057:	ff 74 24 1c          	push   0x1c(%esp)
  40105b:	ff 15 08 60 40 00    	call   *0x406008
  401061:	85 c0                	test   %eax,%eax
  401063:	74 29                	je     0x40108e
  401065:	55                   	push   %ebp
  401066:	8d 44 24 1c          	lea    0x1c(%esp),%eax
  40106a:	50                   	push   %eax
  40106b:	53                   	push   %ebx
  40106c:	6a 02                	push   $0x2
  40106e:	ff 74 24 20          	push   0x20(%esp)
  401072:	ff 15 18 60 40 00    	call   *0x406018
  401078:	85 c0                	test   %eax,%eax
  40107a:	74 12                	je     0x40108e
  40107c:	83 7c 24 18 10       	cmpl   $0x10,0x18(%esp)
  401081:	75 0b                	jne    0x40108e
  401083:	ff 74 24 10          	push   0x10(%esp)
  401087:	46                   	inc    %esi
  401088:	ff 15 0c 60 40 00    	call   *0x40600c
  40108e:	55                   	push   %ebp
  40108f:	ff 74 24 14          	push   0x14(%esp)
  401093:	ff 15 14 60 40 00    	call   *0x406014
  401099:	5f                   	pop    %edi
  40109a:	8b c6                	mov    %esi,%eax
  40109c:	5e                   	pop    %esi
  40109d:	5d                   	pop    %ebp
  40109e:	5b                   	pop    %ebx
  40109f:	83 c4 0c             	add    $0xc,%esp
  4010a2:	c2 04 00             	ret    $0x4
  4010a5:	53                   	push   %ebx
  4010a6:	55                   	push   %ebp
  4010a7:	56                   	push   %esi
  4010a8:	33 f6                	xor    %esi,%esi
  4010aa:	8b da                	mov    %edx,%ebx
  4010ac:	8b e9                	mov    %ecx,%ebp
  4010ae:	39 35 04 ac 41 00    	cmp    %esi,0x41ac04
  4010b4:	75 38                	jne    0x4010ee
  4010b6:	57                   	push   %edi
  4010b7:	8b fe                	mov    %esi,%edi
  4010b9:	b9 00 a8 41 00       	mov    $0x41a800,%ecx
  4010be:	6a 08                	push   $0x8
  4010c0:	8b c7                	mov    %edi,%eax
  4010c2:	5a                   	pop    %edx
  4010c3:	a8 01                	test   $0x1,%al
  4010c5:	74 09                	je     0x4010d0
  4010c7:	d1 e8                	shr    $1,%eax
  4010c9:	35 20 83 b8 ed       	xor    $0xedb88320,%eax
  4010ce:	eb 02                	jmp    0x4010d2
  4010d0:	d1 e8                	shr    $1,%eax
  4010d2:	4a                   	dec    %edx
  4010d3:	75 ee                	jne    0x4010c3
  4010d5:	89 01                	mov    %eax,(%ecx)
  4010d7:	47                   	inc    %edi
  4010d8:	83 c1 04             	add    $0x4,%ecx
  4010db:	81 ff 00 01 00 00    	cmp    $0x100,%edi
  4010e1:	72 db                	jb     0x4010be
  4010e3:	c7 05 04 ac 41 00 01 	movl   $0x1,0x41ac04
  4010ea:	00 00 00 
  4010ed:	5f                   	pop    %edi
  4010ee:	83 c9 ff             	or     $0xffffffff,%ecx
  4010f1:	85 db                	test   %ebx,%ebx
  4010f3:	74 1a                	je     0x40110f
  4010f5:	0f b6 04 2e          	movzbl (%esi,%ebp,1),%eax
  4010f9:	33 c1                	xor    %ecx,%eax
  4010fb:	c1 e9 08             	shr    $0x8,%ecx
  4010fe:	25 ff 00 00 00       	and    $0xff,%eax
  401103:	33 0c 85 00 a8 41 00 	xor    0x41a800(,%eax,4),%ecx
  40110a:	46                   	inc    %esi
  40110b:	3b f3                	cmp    %ebx,%esi
  40110d:	72 e6                	jb     0x4010f5
  40110f:	5e                   	pop    %esi
  401110:	f7 d1                	not    %ecx
  401112:	5d                   	pop    %ebp
  401113:	8b c1                	mov    %ecx,%eax
  401115:	5b                   	pop    %ebx
  401116:	c3                   	ret
  401117:	51                   	push   %ecx
  401118:	51                   	push   %ecx
  401119:	53                   	push   %ebx
  40111a:	55                   	push   %ebp
  40111b:	57                   	push   %edi
  40111c:	8b 7c 24 18          	mov    0x18(%esp),%edi
  401120:	8b c2                	mov    %edx,%eax
  401122:	33 ed                	xor    %ebp,%ebp
  401124:	89 44 24 10          	mov    %eax,0x10(%esp)
  401128:	89 4c 24 0c          	mov    %ecx,0xc(%esp)
  40112c:	8a 9f 00 01 00 00    	mov    0x100(%edi),%bl
  401132:	8a bf 01 01 00 00    	mov    0x101(%edi),%bh
  401138:	85 c0                	test   %eax,%eax
  40113a:	74 38                	je     0x401174
  40113c:	56                   	push   %esi
  40113d:	fe c3                	inc    %bl
  40113f:	0f b6 f3             	movzbl %bl,%esi
  401142:	8a 14 3e             	mov    (%esi,%edi,1),%dl
  401145:	02 fa                	add    %dl,%bh
  401147:	0f b6 cf             	movzbl %bh,%ecx
  40114a:	8a 04 39             	mov    (%ecx,%edi,1),%al
  40114d:	88 04 3e             	mov    %al,(%esi,%edi,1)
  401150:	88 14 39             	mov    %dl,(%ecx,%edi,1)
  401153:	0f b6 0c 3e          	movzbl (%esi,%edi,1),%ecx
  401157:	0f b6 c2             	movzbl %dl,%eax
  40115a:	03 c8                	add    %eax,%ecx
  40115c:	81 e1 ff 00 00 00    	and    $0xff,%ecx
  401162:	8a 04 39             	mov    (%ecx,%edi,1),%al
  401165:	8b 4c 24 10          	mov    0x10(%esp),%ecx
  401169:	30 04 29             	xor    %al,(%ecx,%ebp,1)
  40116c:	45                   	inc    %ebp
  40116d:	3b 6c 24 14          	cmp    0x14(%esp),%ebp
  401171:	72 ca                	jb     0x40113d
  401173:	5e                   	pop    %esi
  401174:	88 9f 00 01 00 00    	mov    %bl,0x100(%edi)
  40117a:	88 bf 01 01 00 00    	mov    %bh,0x101(%edi)
  401180:	5f                   	pop    %edi
  401181:	5d                   	pop    %ebp
  401182:	5b                   	pop    %ebx
  401183:	59                   	pop    %ecx
  401184:	59                   	pop    %ecx
  401185:	c2 04 00             	ret    $0x4
  401188:	55                   	push   %ebp
  401189:	8b ec                	mov    %esp,%ebp
  40118b:	51                   	push   %ecx
  40118c:	56                   	push   %esi
  40118d:	57                   	push   %edi
  40118e:	33 ff                	xor    %edi,%edi
  401190:	8b f2                	mov    %edx,%esi
  401192:	57                   	push   %edi
  401193:	57                   	push   %edi
  401194:	6a 03                	push   $0x3
  401196:	57                   	push   %edi
  401197:	6a 01                	push   $0x1
  401199:	68 00 00 00 80       	push   $0x80000000
  40119e:	51                   	push   %ecx
  40119f:	ff 15 88 60 40 00    	call   *0x406088
  4011a5:	89 46 08             	mov    %eax,0x8(%esi)
  4011a8:	83 f8 ff             	cmp    $0xffffffff,%eax
  4011ab:	74 65                	je     0x401212
  4011ad:	57                   	push   %edi
  4011ae:	50                   	push   %eax
  4011af:	ff 15 98 60 40 00    	call   *0x406098
  4011b5:	83 f8 ff             	cmp    $0xffffffff,%eax
  4011b8:	74 4f                	je     0x401209
  4011ba:	89 46 04             	mov    %eax,0x4(%esi)
  4011bd:	85 c0                	test   %eax,%eax
  4011bf:	75 07                	jne    0x4011c8
  4011c1:	89 3e                	mov    %edi,(%esi)
  4011c3:	33 c0                	xor    %eax,%eax
  4011c5:	40                   	inc    %eax
  4011c6:	eb 4c                	jmp    0x401214
  4011c8:	6a 04                	push   $0x4
  4011ca:	68 00 30 00 00       	push   $0x3000
  4011cf:	50                   	push   %eax
  4011d0:	57                   	push   %edi
  4011d1:	ff 15 84 60 40 00    	call   *0x406084
  4011d7:	89 06                	mov    %eax,(%esi)
  4011d9:	85 c0                	test   %eax,%eax
  4011db:	74 2c                	je     0x401209
  4011dd:	57                   	push   %edi
  4011de:	8d 4d fc             	lea    -0x4(%ebp),%ecx
  4011e1:	51                   	push   %ecx
  4011e2:	ff 76 04             	push   0x4(%esi)
  4011e5:	50                   	push   %eax
  4011e6:	ff 76 08             	push   0x8(%esi)
  4011e9:	ff 15 8c 60 40 00    	call   *0x40608c
  4011ef:	85 c0                	test   %eax,%eax
  4011f1:	74 08                	je     0x4011fb
  4011f3:	8b 45 fc             	mov    -0x4(%ebp),%eax
  4011f6:	3b 46 04             	cmp    0x4(%esi),%eax
  4011f9:	74 c8                	je     0x4011c3
  4011fb:	68 00 80 00 00       	push   $0x8000
  401200:	57                   	push   %edi
  401201:	ff 36                	push   (%esi)
  401203:	ff 15 90 60 40 00    	call   *0x406090
  401209:	ff 76 08             	push   0x8(%esi)
  40120c:	ff 15 80 60 40 00    	call   *0x406080
  401212:	33 c0                	xor    %eax,%eax
  401214:	5f                   	pop    %edi
  401215:	5e                   	pop    %esi
  401216:	8b e5                	mov    %ebp,%esp
  401218:	5d                   	pop    %ebp
  401219:	c2 04 00             	ret    $0x4
  40121c:	55                   	push   %ebp
  40121d:	8d 6c 24 88          	lea    -0x78(%esp),%ebp
  401221:	81 ec d4 00 00 00    	sub    $0xd4,%esp
  401227:	8d 55 64             	lea    0x64(%ebp),%edx
  40122a:	56                   	push   %esi
  40122b:	6a 08                	push   $0x8
  40122d:	59                   	pop    %ecx
  40122e:	e8 21 17 00 00       	call   0x402954
  401233:	6a 09                	push   $0x9
  401235:	8d 55 54             	lea    0x54(%ebp),%edx
  401238:	59                   	pop    %ecx
  401239:	e8 16 17 00 00       	call   0x402954
  40123e:	8b 35 64 60 40 00    	mov    0x406064,%esi
  401244:	8d 45 64             	lea    0x64(%ebp),%eax
  401247:	50                   	push   %eax
  401248:	ff d6                	call   *%esi
  40124a:	a3 2c ac 41 00       	mov    %eax,0x41ac2c
  40124f:	8d 45 54             	lea    0x54(%ebp),%eax
  401252:	50                   	push   %eax
  401253:	ff d6                	call   *%esi
  401255:	6a 13                	push   $0x13
  401257:	8d 55 fc             	lea    -0x4(%ebp),%edx
  40125a:	a3 14 ac 41 00       	mov    %eax,0x41ac14
  40125f:	59                   	pop    %ecx
  401260:	e8 ef 16 00 00       	call   0x402954
  401265:	6a 14                	push   $0x14
  401267:	8d 55 d4             	lea    -0x2c(%ebp),%edx
  40126a:	59                   	pop    %ecx
  40126b:	e8 e4 16 00 00       	call   0x402954
  401270:	6a 15                	push   $0x15
  401272:	8d 55 bc             	lea    -0x44(%ebp),%edx
  401275:	59                   	pop    %ecx
  401276:	e8 d9 16 00 00       	call   0x402954
  40127b:	6a 16                	push   $0x16
  40127d:	8d 55 10             	lea    0x10(%ebp),%edx
  401280:	59                   	pop    %ecx
  401281:	e8 ce 16 00 00       	call   0x402954
  401286:	6a 17                	push   $0x17
  401288:	8d 55 e8             	lea    -0x18(%ebp),%edx
  40128b:	59                   	pop    %ecx
  40128c:	e8 c3 16 00 00       	call   0x402954
  401291:	6a 18                	push   $0x18
  401293:	8d 55 a4             	lea    -0x5c(%ebp),%edx
  401296:	59                   	pop    %ecx
  401297:	e8 b8 16 00 00       	call   0x402954
  40129c:	6a 19                	push   $0x19
  40129e:	8d 55 24             	lea    0x24(%ebp),%edx
  4012a1:	59                   	pop    %ecx
  4012a2:	e8 ad 16 00 00       	call   0x402954
  4012a7:	6a 1a                	push   $0x1a
  4012a9:	8d 55 70             	lea    0x70(%ebp),%edx
  4012ac:	59                   	pop    %ecx
  4012ad:	e8 a2 16 00 00       	call   0x402954
  4012b2:	8b 35 68 60 40 00    	mov    0x406068,%esi
  4012b8:	8d 45 fc             	lea    -0x4(%ebp),%eax
  4012bb:	50                   	push   %eax
  4012bc:	ff 35 2c ac 41 00    	push   0x41ac2c
  4012c2:	ff d6                	call   *%esi
  4012c4:	a3 10 ac 41 00       	mov    %eax,0x41ac10
  4012c9:	8d 45 d4             	lea    -0x2c(%ebp),%eax
  4012cc:	50                   	push   %eax
  4012cd:	ff 35 2c ac 41 00    	push   0x41ac2c
  4012d3:	ff d6                	call   *%esi
  4012d5:	a3 28 ac 41 00       	mov    %eax,0x41ac28
  4012da:	8d 45 bc             	lea    -0x44(%ebp),%eax
  4012dd:	50                   	push   %eax
  4012de:	ff 35 2c ac 41 00    	push   0x41ac2c
  4012e4:	ff d6                	call   *%esi
  4012e6:	a3 18 ac 41 00       	mov    %eax,0x41ac18
  4012eb:	8d 45 10             	lea    0x10(%ebp),%eax
  4012ee:	50                   	push   %eax
  4012ef:	ff 35 2c ac 41 00    	push   0x41ac2c
  4012f5:	ff d6                	call   *%esi
  4012f7:	a3 30 ac 41 00       	mov    %eax,0x41ac30
  4012fc:	8d 45 e8             	lea    -0x18(%ebp),%eax
  4012ff:	50                   	push   %eax
  401300:	ff 35 2c ac 41 00    	push   0x41ac2c
  401306:	ff d6                	call   *%esi
  401308:	a3 0c ac 41 00       	mov    %eax,0x41ac0c
  40130d:	8d 45 a4             	lea    -0x5c(%ebp),%eax
  401310:	50                   	push   %eax
  401311:	ff 35 2c ac 41 00    	push   0x41ac2c
  401317:	ff d6                	call   *%esi
  401319:	a3 20 ac 41 00       	mov    %eax,0x41ac20
  40131e:	8d 45 24             	lea    0x24(%ebp),%eax
  401321:	50                   	push   %eax
  401322:	ff 35 2c ac 41 00    	push   0x41ac2c
  401328:	ff d6                	call   *%esi
  40132a:	a3 08 ac 41 00       	mov    %eax,0x41ac08
  40132f:	8d 45 70             	lea    0x70(%ebp),%eax
  401332:	50                   	push   %eax
  401333:	ff 35 2c ac 41 00    	push   0x41ac2c
  401339:	ff d6                	call   *%esi
  40133b:	6a 1b                	push   $0x1b
  40133d:	8d 55 34             	lea    0x34(%ebp),%edx
  401340:	a3 24 ac 41 00       	mov    %eax,0x41ac24
  401345:	59                   	pop    %ecx
  401346:	e8 09 16 00 00       	call   0x402954
  40134b:	6a 1c                	push   $0x1c
  40134d:	8d 55 44             	lea    0x44(%ebp),%edx
  401350:	59                   	pop    %ecx
  401351:	e8 fe 15 00 00       	call   0x402954
  401356:	8d 45 34             	lea    0x34(%ebp),%eax
  401359:	50                   	push   %eax
  40135a:	ff 35 14 ac 41 00    	push   0x41ac14
  401360:	ff d6                	call   *%esi
  401362:	a3 34 ac 41 00       	mov    %eax,0x41ac34
  401367:	8d 45 44             	lea    0x44(%ebp),%eax
  40136a:	50                   	push   %eax
  40136b:	ff 35 14 ac 41 00    	push   0x41ac14
  401371:	ff d6                	call   *%esi
  401373:	a3 1c ac 41 00       	mov    %eax,0x41ac1c
  401378:	5e                   	pop    %esi
  401379:	8d 65 78             	lea    0x78(%ebp),%esp
  40137c:	5d                   	pop    %ebp
  40137d:	c3                   	ret
  40137e:	55                   	push   %ebp
  40137f:	8b ec                	mov    %esp,%ebp
  401381:	83 ec 44             	sub    $0x44,%esp
  401384:	8b d1                	mov    %ecx,%edx
  401386:	53                   	push   %ebx
  401387:	6a 44                	push   $0x44
  401389:	58                   	pop    %eax
  40138a:	33 db                	xor    %ebx,%ebx
  40138c:	48                   	dec    %eax
  40138d:	88 5c 05 bc          	mov    %bl,-0x44(%ebp,%eax,1)
  401391:	75 f9                	jne    0x40138c
  401393:	6a 10                	push   $0x10
  401395:	58                   	pop    %eax
  401396:	8d 4a 04             	lea    0x4(%edx),%ecx
  401399:	48                   	dec    %eax
  40139a:	88 1c 01             	mov    %bl,(%ecx,%eax,1)
  40139d:	75 fa                	jne    0x401399
  40139f:	51                   	push   %ecx
  4013a0:	8d 45 bc             	lea    -0x44(%ebp),%eax
  4013a3:	50                   	push   %eax
  4013a4:	53                   	push   %ebx
  4013a5:	53                   	push   %ebx
  4013a6:	6a 04                	push   $0x4
  4013a8:	53                   	push   %ebx
  4013a9:	53                   	push   %ebx
  4013aa:	53                   	push   %ebx
  4013ab:	ff 32                	push   (%edx)
  4013ad:	53                   	push   %ebx
  4013ae:	ff 15 1c ac 41 00    	call   *0x41ac1c
  4013b4:	5b                   	pop    %ebx
  4013b5:	8b e5                	mov    %ebp,%esp
  4013b7:	5d                   	pop    %ebp
  4013b8:	c3                   	ret
  4013b9:	81 ec 2c 06 00 00    	sub    $0x62c,%esp
  4013bf:	ff 15 58 60 40 00    	call   *0x406058
  4013c5:	83 25 00 ac 41 00 00 	andl   $0x0,0x41ac00
  4013cc:	8d 54 24 30          	lea    0x30(%esp),%edx
  4013d0:	83 25 04 ac 41 00 00 	andl   $0x0,0x41ac04
  4013d7:	6a 2b                	push   $0x2b
  4013d9:	59                   	pop    %ecx
  4013da:	a3 38 ac 41 00       	mov    %eax,0x41ac38
  4013df:	e8 b8 15 00 00       	call   0x40299c
  4013e4:	8d 44 24 30          	lea    0x30(%esp),%eax
  4013e8:	50                   	push   %eax
  4013e9:	ff 15 50 60 40 00    	call   *0x406050
  4013ef:	85 c0                	test   %eax,%eax
  4013f1:	0f 85 10 03 00 00    	jne    0x401707
  4013f7:	e8 d3 0f 00 00       	call   0x4023cf
  4013fc:	83 24 24 00          	andl   $0x0,(%esp)
  401400:	6a 24                	push   $0x24
  401402:	58                   	pop    %eax
  401403:	48                   	dec    %eax
  401404:	c6 44 04 08 00       	movb   $0x0,0x8(%esp,%eax,1)
  401409:	75 f8                	jne    0x401403
  40140b:	68 04 01 00 00       	push   $0x104
  401410:	8d 84 24 28 04 00 00 	lea    0x428(%esp),%eax
  401417:	50                   	push   %eax
  401418:	6a 00                	push   $0x0
  40141a:	ff 15 6c 60 40 00    	call   *0x40606c
  401420:	51                   	push   %ecx
  401421:	8d 54 24 34          	lea    0x34(%esp),%edx
  401425:	8d 8c 24 28 04 00 00 	lea    0x428(%esp),%ecx
  40142c:	e8 57 fd ff ff       	call   0x401188
  401431:	85 c0                	test   %eax,%eax
  401433:	0f 84 ce 02 00 00    	je     0x401707
  401439:	33 c9                	xor    %ecx,%ecx
  40143b:	8a 81 68 62 40 00    	mov    0x406268(%ecx),%al
  401441:	88 84 0c 14 03 00 00 	mov    %al,0x314(%esp,%ecx,1)
  401448:	41                   	inc    %ecx
  401449:	81 f9 10 01 00 00    	cmp    $0x110,%ecx
  40144f:	72 ea                	jb     0x40143b
  401451:	83 64 24 04 00       	andl   $0x0,0x4(%esp)
  401456:	8d 84 24 14 03 00 00 	lea    0x314(%esp),%eax
  40145d:	53                   	push   %ebx
  40145e:	56                   	push   %esi
  40145f:	8b 74 24 38          	mov    0x38(%esp),%esi
  401463:	8d 54 24 08          	lea    0x8(%esp),%edx
  401467:	50                   	push   %eax
  401468:	ff 74 24 40          	push   0x40(%esp)
  40146c:	8d 4c 24 14          	lea    0x14(%esp),%ecx
  401470:	56                   	push   %esi
  401471:	e8 76 04 00 00       	call   0x4018ec
  401476:	85 c0                	test   %eax,%eax
  401478:	74 06                	je     0x401480
  40147a:	8b 5c 24 0c          	mov    0xc(%esp),%ebx
  40147e:	eb 0b                	jmp    0x40148b
  401480:	8b 4c 24 0c          	mov    0xc(%esp),%ecx
  401484:	e8 9e 02 00 00       	call   0x401727
  401489:	33 db                	xor    %ebx,%ebx
  40148b:	89 5c 24 34          	mov    %ebx,0x34(%esp)
  40148f:	85 f6                	test   %esi,%esi
  401491:	74 0e                	je     0x4014a1
  401493:	68 00 80 00 00       	push   $0x8000
  401498:	6a 00                	push   $0x0
  40149a:	56                   	push   %esi
  40149b:	ff 15 90 60 40 00    	call   *0x406090
  4014a1:	83 7c 24 40 00       	cmpl   $0x0,0x40(%esp)
  4014a6:	74 0a                	je     0x4014b2
  4014a8:	ff 74 24 40          	push   0x40(%esp)
  4014ac:	ff 15 80 60 40 00    	call   *0x406080
  4014b2:	85 db                	test   %ebx,%ebx
  4014b4:	0f 84 4b 02 00 00    	je     0x401705
  4014ba:	83 7c 24 08 00       	cmpl   $0x0,0x8(%esp)
  4014bf:	0f 86 40 02 00 00    	jbe    0x401705
  4014c5:	51                   	push   %ecx
  4014c6:	ff 74 24 0c          	push   0xc(%esp)
  4014ca:	8b d3                	mov    %ebx,%edx
  4014cc:	8d 4c 24 10          	lea    0x10(%esp),%ecx
  4014d0:	e8 99 12 00 00       	call   0x40276e
  4014d5:	83 7c 24 08 00       	cmpl   $0x0,0x8(%esp)
  4014da:	0f 84 1e 02 00 00    	je     0x4016fe
  4014e0:	55                   	push   %ebp
  4014e1:	57                   	push   %edi
  4014e2:	51                   	push   %ecx
  4014e3:	8b 4c 24 14          	mov    0x14(%esp),%ecx
  4014e7:	33 ed                	xor    %ebp,%ebp
  4014e9:	33 f6                	xor    %esi,%esi
  4014eb:	e8 03 12 00 00       	call   0x4026f3
  4014f0:	8b f8                	mov    %eax,%edi
  4014f2:	85 ff                	test   %edi,%edi
  4014f4:	74 10                	je     0x401506
  4014f6:	8b cf                	mov    %edi,%ecx
  4014f8:	e8 1c 12 00 00       	call   0x402719
  4014fd:	8b f0                	mov    %eax,%esi
  4014ff:	85 f6                	test   %esi,%esi
  401501:	74 03                	je     0x401506
  401503:	8b 6f 0c             	mov    0xc(%edi),%ebp
  401506:	89 74 24 38          	mov    %esi,0x38(%esp)
  40150a:	85 ed                	test   %ebp,%ebp
  40150c:	0f 84 e1 01 00 00    	je     0x4016f3
  401512:	85 f6                	test   %esi,%esi
  401514:	0f 84 d9 01 00 00    	je     0x4016f3
  40151a:	e8 fd fc ff ff       	call   0x40121c
  40151f:	ff 15 74 60 40 00    	call   *0x406074
  401525:	8b c8                	mov    %eax,%ecx
  401527:	e8 4a 13 00 00       	call   0x402876
  40152c:	89 44 24 18          	mov    %eax,0x18(%esp)
  401530:	85 c0                	test   %eax,%eax
  401532:	0f 84 b2 01 00 00    	je     0x4016ea
  401538:	33 db                	xor    %ebx,%ebx
  40153a:	33 f6                	xor    %esi,%esi
  40153c:	89 5c 24 14          	mov    %ebx,0x14(%esp)
  401540:	8d 4c 24 18          	lea    0x18(%esp),%ecx
  401544:	c7 44 24 58 07 00 01 	movl   $0x10007,0x58(%esp)
  40154b:	00 
  40154c:	46                   	inc    %esi
  40154d:	e8 2c fe ff ff       	call   0x40137e
  401552:	85 c0                	test   %eax,%eax
  401554:	0f 84 6d 01 00 00    	je     0x4016c7
  40155a:	8b 4c 24 38          	mov    0x38(%esp),%ecx
  40155e:	b8 4d 5a 00 00       	mov    $0x5a4d,%eax
  401563:	89 4c 24 2c          	mov    %ecx,0x2c(%esp)
  401567:	66 39 01             	cmp    %ax,(%ecx)
  40156a:	0f 85 33 01 00 00    	jne    0x4016a3
  401570:	8b 41 3c             	mov    0x3c(%ecx),%eax
  401573:	03 c1                	add    %ecx,%eax
  401575:	89 44 24 30          	mov    %eax,0x30(%esp)
  401579:	8d 44 24 58          	lea    0x58(%esp),%eax
  40157d:	50                   	push   %eax
  40157e:	ff 74 24 24          	push   0x24(%esp)
  401582:	ff 15 30 ac 41 00    	call   *0x41ac30
  401588:	53                   	push   %ebx
  401589:	6a 04                	push   $0x4
  40158b:	8d 44 24 1c          	lea    0x1c(%esp),%eax
  40158f:	50                   	push   %eax
  401590:	8b 84 24 08 01 00 00 	mov    0x108(%esp),%eax
  401597:	83 c0 08             	add    $0x8,%eax
  40159a:	50                   	push   %eax
  40159b:	ff 74 24 2c          	push   0x2c(%esp)
  40159f:	ff 15 28 ac 41 00    	call   *0x41ac28
  4015a5:	8b 44 24 30          	mov    0x30(%esp),%eax
  4015a9:	8b 4c 24 14          	mov    0x14(%esp),%ecx
  4015ad:	3b 48 34             	cmp    0x34(%eax),%ecx
  4015b0:	75 0f                	jne    0x4015c1
  4015b2:	51                   	push   %ecx
  4015b3:	ff 74 24 20          	push   0x20(%esp)
  4015b7:	ff 15 20 ac 41 00    	call   *0x41ac20
  4015bd:	8b 44 24 30          	mov    0x30(%esp),%eax
  4015c1:	6a 40                	push   $0x40
  4015c3:	68 00 30 00 00       	push   $0x3000
  4015c8:	ff 70 50             	push   0x50(%eax)
  4015cb:	ff 70 34             	push   0x34(%eax)
  4015ce:	ff 74 24 2c          	push   0x2c(%esp)
  4015d2:	ff 15 34 ac 41 00    	call   *0x41ac34
  4015d8:	8b f8                	mov    %eax,%edi
  4015da:	85 ff                	test   %edi,%edi
  4015dc:	0f 84 c1 00 00 00    	je     0x4016a3
  4015e2:	8b 44 24 30          	mov    0x30(%esp),%eax
  4015e6:	53                   	push   %ebx
  4015e7:	ff 70 54             	push   0x54(%eax)
  4015ea:	ff 74 24 40          	push   0x40(%esp)
  4015ee:	57                   	push   %edi
  4015ef:	ff 74 24 2c          	push   0x2c(%esp)
  4015f3:	ff 15 18 ac 41 00    	call   *0x41ac18
  4015f9:	85 c0                	test   %eax,%eax
  4015fb:	0f 88 a2 00 00 00    	js     0x4016a3
  401601:	8b 4c 24 30          	mov    0x30(%esp),%ecx
  401605:	33 c0                	xor    %eax,%eax
  401607:	66 3b 41 06          	cmp    0x6(%ecx),%ax
  40160b:	73 46                	jae    0x401653
  40160d:	33 ed                	xor    %ebp,%ebp
  40160f:	8b 44 24 2c          	mov    0x2c(%esp),%eax
  401613:	8b 4c 24 38          	mov    0x38(%esp),%ecx
  401617:	6a 00                	push   $0x0
  401619:	8b 40 3c             	mov    0x3c(%eax),%eax
  40161c:	03 c5                	add    %ebp,%eax
  40161e:	8d 91 f8 00 00 00    	lea    0xf8(%ecx),%edx
  401624:	03 d0                	add    %eax,%edx
  401626:	89 54 24 38          	mov    %edx,0x38(%esp)
  40162a:	ff 72 10             	push   0x10(%edx)
  40162d:	8b 42 14             	mov    0x14(%edx),%eax
  401630:	03 c1                	add    %ecx,%eax
  401632:	50                   	push   %eax
  401633:	8b 42 0c             	mov    0xc(%edx),%eax
  401636:	03 c7                	add    %edi,%eax
  401638:	50                   	push   %eax
  401639:	ff 74 24 2c          	push   0x2c(%esp)
  40163d:	ff 15 18 ac 41 00    	call   *0x41ac18
  401643:	8b 4c 24 30          	mov    0x30(%esp),%ecx
  401647:	43                   	inc    %ebx
  401648:	83 c5 28             	add    $0x28,%ebp
  40164b:	0f b7 41 06          	movzwl 0x6(%ecx),%eax
  40164f:	3b d8                	cmp    %eax,%ebx
  401651:	72 bc                	jb     0x40160f
  401653:	8b 41 28             	mov    0x28(%ecx),%eax
  401656:	33 db                	xor    %ebx,%ebx
  401658:	03 c7                	add    %edi,%eax
  40165a:	53                   	push   %ebx
  40165b:	89 84 24 0c 01 00 00 	mov    %eax,0x10c(%esp)
  401662:	8d 41 34             	lea    0x34(%ecx),%eax
  401665:	6a 04                	push   $0x4
  401667:	50                   	push   %eax
  401668:	8b 84 24 08 01 00 00 	mov    0x108(%esp),%eax
  40166f:	83 c0 08             	add    $0x8,%eax
  401672:	50                   	push   %eax
  401673:	ff 74 24 2c          	push   0x2c(%esp)
  401677:	ff 15 18 ac 41 00    	call   *0x41ac18
  40167d:	85 c0                	test   %eax,%eax
  40167f:	78 22                	js     0x4016a3
  401681:	8d 44 24 58          	lea    0x58(%esp),%eax
  401685:	50                   	push   %eax
  401686:	ff 74 24 24          	push   0x24(%esp)
  40168a:	ff 15 0c ac 41 00    	call   *0x41ac0c
  401690:	85 c0                	test   %eax,%eax
  401692:	78 0f                	js     0x4016a3
  401694:	53                   	push   %ebx
  401695:	ff 74 24 24          	push   0x24(%esp)
  401699:	ff 15 08 ac 41 00    	call   *0x41ac08
  40169f:	85 c0                	test   %eax,%eax
  4016a1:	79 0e                	jns    0x4016b1
  4016a3:	6a 01                	push   $0x1
  4016a5:	ff 74 24 20          	push   0x20(%esp)
  4016a9:	ff 15 10 ac 41 00    	call   *0x41ac10
  4016af:	8b f3                	mov    %ebx,%esi
  4016b1:	ff 74 24 20          	push   0x20(%esp)
  4016b5:	ff 15 24 ac 41 00    	call   *0x41ac24
  4016bb:	ff 74 24 1c          	push   0x1c(%esp)
  4016bf:	ff 15 24 ac 41 00    	call   *0x41ac24
  4016c5:	eb 02                	jmp    0x4016c9
  4016c7:	8b f3                	mov    %ebx,%esi
  4016c9:	85 f6                	test   %esi,%esi
  4016cb:	75 10                	jne    0x4016dd
  4016cd:	68 e8 03 00 00       	push   $0x3e8
  4016d2:	ff 15 70 60 40 00    	call   *0x406070
  4016d8:	e9 5d fe ff ff       	jmp    0x40153a
  4016dd:	8b 4c 24 18          	mov    0x18(%esp),%ecx
  4016e1:	e8 41 00 00 00       	call   0x401727
  4016e6:	8b 5c 24 3c          	mov    0x3c(%esp),%ebx
  4016ea:	8b 4c 24 38          	mov    0x38(%esp),%ecx
  4016ee:	e8 34 00 00 00       	call   0x401727
  4016f3:	8b 4c 24 10          	mov    0x10(%esp),%ecx
  4016f7:	e8 2b 00 00 00       	call   0x401727
  4016fc:	5f                   	pop    %edi
  4016fd:	5d                   	pop    %ebp
  4016fe:	8b cb                	mov    %ebx,%ecx
  401700:	e8 22 00 00 00       	call   0x401727
  401705:	5e                   	pop    %esi
  401706:	5b                   	pop    %ebx
  401707:	6a 00                	push   $0x0
  401709:	ff 15 78 60 40 00    	call   *0x406078
  40170f:	cc                   	int3
  401710:	85 c9                	test   %ecx,%ecx
  401712:	75 03                	jne    0x401717
  401714:	33 c0                	xor    %eax,%eax
  401716:	c3                   	ret
  401717:	51                   	push   %ecx
  401718:	6a 08                	push   $0x8
  40171a:	ff 35 38 ac 41 00    	push   0x41ac38
  401720:	ff 15 60 60 40 00    	call   *0x406060
  401726:	c3                   	ret
  401727:	51                   	push   %ecx
  401728:	6a 00                	push   $0x0
  40172a:	ff 35 38 ac 41 00    	push   0x41ac38
  401730:	ff 15 5c 60 40 00    	call   *0x40605c
  401736:	c3                   	ret
  401737:	56                   	push   %esi
  401738:	33 f6                	xor    %esi,%esi
  40173a:	39 74 24 08          	cmp    %esi,0x8(%esp)
  40173e:	76 0d                	jbe    0x40174d
  401740:	8a 04 16             	mov    (%esi,%edx,1),%al
  401743:	88 04 0e             	mov    %al,(%esi,%ecx,1)
  401746:	46                   	inc    %esi
  401747:	3b 74 24 08          	cmp    0x8(%esp),%esi
  40174b:	72 f3                	jb     0x401740
  40174d:	5e                   	pop    %esi
  40174e:	c2 04 00             	ret    $0x4
  401751:	53                   	push   %ebx
  401752:	56                   	push   %esi
  401753:	8b f1                	mov    %ecx,%esi
  401755:	33 c0                	xor    %eax,%eax
  401757:	2b f2                	sub    %edx,%esi
  401759:	8a 1c 16             	mov    (%esi,%edx,1),%bl
  40175c:	8a 0a                	mov    (%edx),%cl
  40175e:	3a d9                	cmp    %cl,%bl
  401760:	75 0b                	jne    0x40176d
  401762:	40                   	inc    %eax
  401763:	42                   	inc    %edx
  401764:	83 f8 10             	cmp    $0x10,%eax
  401767:	72 f0                	jb     0x401759
  401769:	33 c0                	xor    %eax,%eax
  40176b:	eb 08                	jmp    0x401775
  40176d:	0f b6 c9             	movzbl %cl,%ecx
  401770:	0f b6 c3             	movzbl %bl,%eax
  401773:	2b c1                	sub    %ecx,%eax
  401775:	5e                   	pop    %esi
  401776:	5b                   	pop    %ebx
  401777:	c2 04 00             	ret    $0x4
  40177a:	55                   	push   %ebp
  40177b:	8d 6c 24 88          	lea    -0x78(%esp),%ebp
  40177f:	81 ec 1c 01 00 00    	sub    $0x11c,%esp
  401785:	56                   	push   %esi
  401786:	68 80 69 40 00       	push   $0x406980
  40178b:	68 90 69 40 00       	push   $0x406990
  401790:	ff 15 64 60 40 00    	call   *0x406064
  401796:	50                   	push   %eax
  401797:	ff 15 68 60 40 00    	call   *0x406068
  40179d:	33 f6                	xor    %esi,%esi
  40179f:	85 c0                	test   %eax,%eax
  4017a1:	74 13                	je     0x4017b6
  4017a3:	8d 8d 5c ff ff ff    	lea    -0xa4(%ebp),%ecx
  4017a9:	c7 85 5c ff ff ff 1c 	movl   $0x11c,-0xa4(%ebp)
  4017b0:	01 00 00 
  4017b3:	51                   	push   %ecx
  4017b4:	ff d0                	call   *%eax
  4017b6:	8a 45 76             	mov    0x76(%ebp),%al
  4017b9:	3c 01                	cmp    $0x1,%al
  4017bb:	75 77                	jne    0x401834
  4017bd:	83 bd 60 ff ff ff 05 	cmpl   $0x5,-0xa0(%ebp)
  4017c4:	75 27                	jne    0x4017ed
  4017c6:	8b 85 64 ff ff ff    	mov    -0x9c(%ebp),%eax
  4017cc:	85 c0                	test   %eax,%eax
  4017ce:	75 08                	jne    0x4017d8
  4017d0:	33 f6                	xor    %esi,%esi
  4017d2:	46                   	inc    %esi
  4017d3:	e9 c2 00 00 00       	jmp    0x40189a
  4017d8:	83 f8 01             	cmp    $0x1,%eax
  4017db:	74 09                	je     0x4017e6
  4017dd:	83 f8 02             	cmp    $0x2,%eax
  4017e0:	0f 85 b4 00 00 00    	jne    0x40189a
  4017e6:	6a 02                	push   $0x2
  4017e8:	e9 95 00 00 00       	jmp    0x401882
  4017ed:	83 bd 60 ff ff ff 06 	cmpl   $0x6,-0xa0(%ebp)
  4017f4:	75 29                	jne    0x40181f
  4017f6:	8b 85 64 ff ff ff    	mov    -0x9c(%ebp),%eax
  4017fc:	85 c0                	test   %eax,%eax
  4017fe:	75 04                	jne    0x401804
  401800:	6a 04                	push   $0x4
  401802:	eb 7e                	jmp    0x401882
  401804:	83 f8 01             	cmp    $0x1,%eax
  401807:	75 04                	jne    0x40180d
  401809:	6a 07                	push   $0x7
  40180b:	eb 75                	jmp    0x401882
  40180d:	83 f8 02             	cmp    $0x2,%eax
  401810:	75 04                	jne    0x401816
  401812:	6a 09                	push   $0x9
  401814:	eb 6c                	jmp    0x401882
  401816:	83 f8 03             	cmp    $0x3,%eax
  401819:	75 7f                	jne    0x40189a
  40181b:	6a 0b                	push   $0xb
  40181d:	eb 63                	jmp    0x401882
  40181f:	83 bd 60 ff ff ff 0a 	cmpl   $0xa,-0xa0(%ebp)
  401826:	75 72                	jne    0x40189a
  401828:	39 b5 64 ff ff ff    	cmp    %esi,-0x9c(%ebp)
  40182e:	75 6a                	jne    0x40189a
  401830:	6a 0d                	push   $0xd
  401832:	eb 4e                	jmp    0x401882
  401834:	3c 02                	cmp    $0x2,%al
  401836:	74 04                	je     0x40183c
  401838:	3c 03                	cmp    $0x3,%al
  40183a:	75 5e                	jne    0x40189a
  40183c:	83 bd 60 ff ff ff 05 	cmpl   $0x5,-0xa0(%ebp)
  401843:	75 0d                	jne    0x401852
  401845:	83 bd 64 ff ff ff 02 	cmpl   $0x2,-0x9c(%ebp)
  40184c:	75 4c                	jne    0x40189a
  40184e:	6a 03                	push   $0x3
  401850:	eb 30                	jmp    0x401882
  401852:	83 bd 60 ff ff ff 06 	cmpl   $0x6,-0xa0(%ebp)
  401859:	75 2a                	jne    0x401885
  40185b:	8b 85 64 ff ff ff    	mov    -0x9c(%ebp),%eax
  401861:	85 c0                	test   %eax,%eax
  401863:	75 04                	jne    0x401869
  401865:	6a 05                	push   $0x5
  401867:	eb 19                	jmp    0x401882
  401869:	83 f8 01             	cmp    $0x1,%eax
  40186c:	75 04                	jne    0x401872
  40186e:	6a 06                	push   $0x6
  401870:	eb 10                	jmp    0x401882
  401872:	83 f8 02             	cmp    $0x2,%eax
  401875:	75 04                	jne    0x40187b
  401877:	6a 08                	push   $0x8
  401879:	eb 07                	jmp    0x401882
  40187b:	83 f8 03             	cmp    $0x3,%eax
  40187e:	75 1a                	jne    0x40189a
  401880:	6a 0a                	push   $0xa
  401882:	5e                   	pop    %esi
  401883:	eb 15                	jmp    0x40189a
  401885:	83 bd 60 ff ff ff 0a 	cmpl   $0xa,-0xa0(%ebp)
  40188c:	75 0c                	jne    0x40189a
  40188e:	39 b5 64 ff ff ff    	cmp    %esi,-0x9c(%ebp)
  401894:	6a 0c                	push   $0xc
  401896:	58                   	pop    %eax
  401897:	0f 44 f0             	cmove  %eax,%esi
  40189a:	8b c6                	mov    %esi,%eax
  40189c:	5e                   	pop    %esi
  40189d:	8d 65 78             	lea    0x78(%ebp),%esp
  4018a0:	5d                   	pop    %ebp
  4018a1:	c3                   	ret
  4018a2:	57                   	push   %edi
  4018a3:	8d 79 f4             	lea    -0xc(%ecx),%edi
  4018a6:	03 fa                	add    %edx,%edi
  4018a8:	8b c7                	mov    %edi,%eax
  4018aa:	8d 4f 0c             	lea    0xc(%edi),%ecx
  4018ad:	3b f9                	cmp    %ecx,%edi
  4018af:	77 0d                	ja     0x4018be
  4018b1:	81 38 12 5e d9 57    	cmpl   $0x57d95e12,(%eax)
  4018b7:	74 19                	je     0x4018d2
  4018b9:	40                   	inc    %eax
  4018ba:	3b c1                	cmp    %ecx,%eax
  4018bc:	76 f3                	jbe    0x4018b1
  4018be:	8b 44 24 08          	mov    0x8(%esp),%eax
  4018c2:	83 20 00             	andl   $0x0,(%eax)
  4018c5:	8b 44 24 0c          	mov    0xc(%esp),%eax
  4018c9:	83 20 00             	andl   $0x0,(%eax)
  4018cc:	33 c0                	xor    %eax,%eax
  4018ce:	5f                   	pop    %edi
  4018cf:	c2 08 00             	ret    $0x8
  4018d2:	8b 4c 24 08          	mov    0x8(%esp),%ecx
  4018d6:	56                   	push   %esi
  4018d7:	8b 77 08             	mov    0x8(%edi),%esi
  4018da:	2b fe                	sub    %esi,%edi
  4018dc:	8b c7                	mov    %edi,%eax
  4018de:	8d 56 0c             	lea    0xc(%esi),%edx
  4018e1:	89 11                	mov    %edx,(%ecx)
  4018e3:	8b 4c 24 10          	mov    0x10(%esp),%ecx
  4018e7:	89 31                	mov    %esi,(%ecx)
  4018e9:	5e                   	pop    %esi
  4018ea:	eb e2                	jmp    0x4018ce
  4018ec:	81 ec 0c 01 00 00    	sub    $0x10c,%esp
  4018f2:	53                   	push   %ebx
  4018f3:	55                   	push   %ebp
  4018f4:	56                   	push   %esi
  4018f5:	57                   	push   %edi
  4018f6:	8d 44 24 10          	lea    0x10(%esp),%eax
  4018fa:	8b da                	mov    %edx,%ebx
  4018fc:	8b 94 24 24 01 00 00 	mov    0x124(%esp),%edx
  401903:	8b e9                	mov    %ecx,%ebp
  401905:	8b 8c 24 20 01 00 00 	mov    0x120(%esp),%ecx
  40190c:	50                   	push   %eax
  40190d:	8d 44 24 18          	lea    0x18(%esp),%eax
  401911:	50                   	push   %eax
  401912:	e8 8b ff ff ff       	call   0x4018a2
  401917:	8b 74 24 10          	mov    0x10(%esp),%esi
  40191b:	8b f8                	mov    %eax,%edi
  40191d:	85 ff                	test   %edi,%edi
  40191f:	74 68                	je     0x401989
  401921:	85 f6                	test   %esi,%esi
  401923:	74 5b                	je     0x401980
  401925:	8b 54 3e 08          	mov    0x8(%esi,%edi,1),%edx
  401929:	8b cf                	mov    %edi,%ecx
  40192b:	e8 75 f7 ff ff       	call   0x4010a5
  401930:	39 44 3e 04          	cmp    %eax,0x4(%esi,%edi,1)
  401934:	75 53                	jne    0x401989
  401936:	8b 94 24 28 01 00 00 	mov    0x128(%esp),%edx
  40193d:	85 d2                	test   %edx,%edx
  40193f:	74 20                	je     0x401961
  401941:	33 c9                	xor    %ecx,%ecx
  401943:	8a 04 11             	mov    (%ecx,%edx,1),%al
  401946:	88 44 0c 18          	mov    %al,0x18(%esp,%ecx,1)
  40194a:	41                   	inc    %ecx
  40194b:	81 f9 02 01 00 00    	cmp    $0x102,%ecx
  401951:	72 f0                	jb     0x401943
  401953:	8d 44 24 18          	lea    0x18(%esp),%eax
  401957:	8b d6                	mov    %esi,%edx
  401959:	50                   	push   %eax
  40195a:	8b cf                	mov    %edi,%ecx
  40195c:	e8 b6 f7 ff ff       	call   0x401117
  401961:	8b ce                	mov    %esi,%ecx
  401963:	e8 a8 fd ff ff       	call   0x401710
  401968:	8b c8                	mov    %eax,%ecx
  40196a:	85 c9                	test   %ecx,%ecx
  40196c:	74 1b                	je     0x401989
  40196e:	56                   	push   %esi
  40196f:	8b d7                	mov    %edi,%edx
  401971:	e8 c1 fd ff ff       	call   0x401737
  401976:	89 33                	mov    %esi,(%ebx)
  401978:	89 4d 00             	mov    %ecx,0x0(%ebp)
  40197b:	33 c0                	xor    %eax,%eax
  40197d:	40                   	inc    %eax
  40197e:	eb 0b                	jmp    0x40198b
  401980:	83 65 00 00          	andl   $0x0,0x0(%ebp)
  401984:	83 23 00             	andl   $0x0,(%ebx)
  401987:	eb f2                	jmp    0x40197b
  401989:	33 c0                	xor    %eax,%eax
  40198b:	5f                   	pop    %edi
  40198c:	5e                   	pop    %esi
  40198d:	5d                   	pop    %ebp
  40198e:	5b                   	pop    %ebx
  40198f:	81 c4 0c 01 00 00    	add    $0x10c,%esp
  401995:	c2 0c 00             	ret    $0xc
  401998:	81 ec 30 02 00 00    	sub    $0x230,%esp
  40199e:	55                   	push   %ebp
  40199f:	56                   	push   %esi
  4019a0:	57                   	push   %edi
  4019a1:	33 ff                	xor    %edi,%edi
  4019a3:	89 54 24 0c          	mov    %edx,0xc(%esp)
  4019a7:	57                   	push   %edi
  4019a8:	6a 02                	push   $0x2
  4019aa:	8b e9                	mov    %ecx,%ebp
  4019ac:	ff 15 24 60 40 00    	call   *0x406024
  4019b2:	8b f0                	mov    %eax,%esi
  4019b4:	c7 44 24 10 2c 02 00 	movl   $0x22c,0x10(%esp)
  4019bb:	00 
  4019bc:	8d 44 24 10          	lea    0x10(%esp),%eax
  4019c0:	50                   	push   %eax
  4019c1:	56                   	push   %esi
  4019c2:	ff 15 3c 60 40 00    	call   *0x40603c
  4019c8:	85 c0                	test   %eax,%eax
  4019ca:	74 30                	je     0x4019fc
  4019cc:	eb 19                	jmp    0x4019e7
  4019ce:	39 6c 24 18          	cmp    %ebp,0x18(%esp)
  4019d2:	75 13                	jne    0x4019e7
  4019d4:	51                   	push   %ecx
  4019d5:	51                   	push   %ecx
  4019d6:	8b 4c 24 14          	mov    0x14(%esp),%ecx
  4019da:	8d 54 24 3c          	lea    0x3c(%esp),%edx
  4019de:	e8 12 0e 00 00       	call   0x4027f5
  4019e3:	85 c0                	test   %eax,%eax
  4019e5:	74 12                	je     0x4019f9
  4019e7:	8d 44 24 10          	lea    0x10(%esp),%eax
  4019eb:	50                   	push   %eax
  4019ec:	56                   	push   %esi
  4019ed:	ff 15 20 60 40 00    	call   *0x406020
  4019f3:	85 c0                	test   %eax,%eax
  4019f5:	75 d7                	jne    0x4019ce
  4019f7:	eb 03                	jmp    0x4019fc
  4019f9:	33 ff                	xor    %edi,%edi
  4019fb:	47                   	inc    %edi
  4019fc:	56                   	push   %esi
  4019fd:	ff 15 80 60 40 00    	call   *0x406080
  401a03:	8b c7                	mov    %edi,%eax
  401a05:	5f                   	pop    %edi
  401a06:	5e                   	pop    %esi
  401a07:	5d                   	pop    %ebp
  401a08:	81 c4 30 02 00 00    	add    $0x230,%esp
  401a0e:	c3                   	ret
  401a0f:	81 ec 2c 02 00 00    	sub    $0x22c,%esp
  401a15:	55                   	push   %ebp
  401a16:	56                   	push   %esi
  401a17:	57                   	push   %edi
  401a18:	33 ff                	xor    %edi,%edi
  401a1a:	8b e9                	mov    %ecx,%ebp
  401a1c:	57                   	push   %edi
  401a1d:	6a 02                	push   $0x2
  401a1f:	ff 15 24 60 40 00    	call   *0x406024
  401a25:	8b f0                	mov    %eax,%esi
  401a27:	c7 44 24 0c 2c 02 00 	movl   $0x22c,0xc(%esp)
  401a2e:	00 
  401a2f:	8d 44 24 0c          	lea    0xc(%esp),%eax
  401a33:	50                   	push   %eax
  401a34:	56                   	push   %esi
  401a35:	ff 15 3c 60 40 00    	call   *0x40603c
  401a3b:	85 c0                	test   %eax,%eax
  401a3d:	74 28                	je     0x401a67
  401a3f:	eb 11                	jmp    0x401a52
  401a41:	51                   	push   %ecx
  401a42:	51                   	push   %ecx
  401a43:	8d 54 24 38          	lea    0x38(%esp),%edx
  401a47:	8b cd                	mov    %ebp,%ecx
  401a49:	e8 a7 0d 00 00       	call   0x4027f5
  401a4e:	85 c0                	test   %eax,%eax
  401a50:	74 12                	je     0x401a64
  401a52:	8d 44 24 0c          	lea    0xc(%esp),%eax
  401a56:	50                   	push   %eax
  401a57:	56                   	push   %esi
  401a58:	ff 15 20 60 40 00    	call   *0x406020
  401a5e:	85 c0                	test   %eax,%eax
  401a60:	75 df                	jne    0x401a41
  401a62:	eb 03                	jmp    0x401a67
  401a64:	33 ff                	xor    %edi,%edi
  401a66:	47                   	inc    %edi
  401a67:	56                   	push   %esi
  401a68:	ff 15 80 60 40 00    	call   *0x406080
  401a6e:	8b c7                	mov    %edi,%eax
  401a70:	5f                   	pop    %edi
  401a71:	5e                   	pop    %esi
  401a72:	5d                   	pop    %ebp
  401a73:	81 c4 2c 02 00 00    	add    $0x22c,%esp
  401a79:	c3                   	ret
  401a7a:	55                   	push   %ebp
  401a7b:	8d 6c 24 88          	lea    -0x78(%esp),%ebp
  401a7f:	81 ec e0 00 00 00    	sub    $0xe0,%esp
  401a85:	56                   	push   %esi
  401a86:	6a 2e                	push   $0x2e
  401a88:	8d 55 e4             	lea    -0x1c(%ebp),%edx
  401a8b:	59                   	pop    %ecx
  401a8c:	e8 0b 0f 00 00       	call   0x40299c
  401a91:	6a 2f                	push   $0x2f
  401a93:	8d 55 cc             	lea    -0x34(%ebp),%edx
  401a96:	59                   	pop    %ecx
  401a97:	e8 00 0f 00 00       	call   0x40299c
  401a9c:	6a 30                	push   $0x30
  401a9e:	8d 55 44             	lea    0x44(%ebp),%edx
  401aa1:	59                   	pop    %ecx
  401aa2:	e8 f5 0e 00 00       	call   0x40299c
  401aa7:	6a 31                	push   $0x31
  401aa9:	8d 55 14             	lea    0x14(%ebp),%edx
  401aac:	59                   	pop    %ecx
  401aad:	e8 ea 0e 00 00       	call   0x40299c
  401ab2:	6a 32                	push   $0x32
  401ab4:	8d 55 98             	lea    -0x68(%ebp),%edx
  401ab7:	59                   	pop    %ecx
  401ab8:	e8 df 0e 00 00       	call   0x40299c
  401abd:	6a 33                	push   $0x33
  401abf:	8d 55 b4             	lea    -0x4c(%ebp),%edx
  401ac2:	59                   	pop    %ecx
  401ac3:	e8 d4 0e 00 00       	call   0x40299c
  401ac8:	6a 34                	push   $0x34
  401aca:	8d 55 2c             	lea    0x2c(%ebp),%edx
  401acd:	59                   	pop    %ecx
  401ace:	e8 c9 0e 00 00       	call   0x40299c
  401ad3:	6a 35                	push   $0x35
  401ad5:	8d 55 fc             	lea    -0x4(%ebp),%edx
  401ad8:	59                   	pop    %ecx
  401ad9:	e8 be 0e 00 00       	call   0x40299c
  401ade:	8d 45 e4             	lea    -0x1c(%ebp),%eax
  401ae1:	33 f6                	xor    %esi,%esi
  401ae3:	89 45 58             	mov    %eax,0x58(%ebp)
  401ae6:	8d 45 cc             	lea    -0x34(%ebp),%eax
  401ae9:	89 45 5c             	mov    %eax,0x5c(%ebp)
  401aec:	8d 45 44             	lea    0x44(%ebp),%eax
  401aef:	89 45 60             	mov    %eax,0x60(%ebp)
  401af2:	8d 45 14             	lea    0x14(%ebp),%eax
  401af5:	89 45 64             	mov    %eax,0x64(%ebp)
  401af8:	8d 45 98             	lea    -0x68(%ebp),%eax
  401afb:	89 45 68             	mov    %eax,0x68(%ebp)
  401afe:	8d 45 b4             	lea    -0x4c(%ebp),%eax
  401b01:	89 45 6c             	mov    %eax,0x6c(%ebp)
  401b04:	8d 45 2c             	lea    0x2c(%ebp),%eax
  401b07:	89 45 70             	mov    %eax,0x70(%ebp)
  401b0a:	8d 45 fc             	lea    -0x4(%ebp),%eax
  401b0d:	89 45 74             	mov    %eax,0x74(%ebp)
  401b10:	ff 74 b5 58          	push   0x58(%ebp,%esi,4)
  401b14:	ff 15 50 60 40 00    	call   *0x406050
  401b1a:	85 c0                	test   %eax,%eax
  401b1c:	75 08                	jne    0x401b26
  401b1e:	46                   	inc    %esi
  401b1f:	83 fe 08             	cmp    $0x8,%esi
  401b22:	72 ec                	jb     0x401b10
  401b24:	eb 03                	jmp    0x401b29
  401b26:	33 c0                	xor    %eax,%eax
  401b28:	40                   	inc    %eax
  401b29:	5e                   	pop    %esi
  401b2a:	8d 65 78             	lea    0x78(%ebp),%esp
  401b2d:	5d                   	pop    %ebp
  401b2e:	c3                   	ret
  401b2f:	55                   	push   %ebp
  401b30:	8d 6c 24 88          	lea    -0x78(%esp),%ebp
  401b34:	81 ec 30 03 00 00    	sub    $0x330,%esp
  401b3a:	56                   	push   %esi
  401b3b:	6a 59                	push   $0x59
  401b3d:	8d 95 f8 fe ff ff    	lea    -0x108(%ebp),%edx
  401b43:	59                   	pop    %ecx
  401b44:	e8 53 0e 00 00       	call   0x40299c
  401b49:	6a 5a                	push   $0x5a
  401b4b:	8d 95 98 fd ff ff    	lea    -0x268(%ebp),%edx
  401b51:	59                   	pop    %ecx
  401b52:	e8 45 0e 00 00       	call   0x40299c
  401b57:	6a 5b                	push   $0x5b
  401b59:	8d 55 88             	lea    -0x78(%ebp),%edx
  401b5c:	59                   	pop    %ecx
  401b5d:	e8 3a 0e 00 00       	call   0x40299c
  401b62:	6a 5c                	push   $0x5c
  401b64:	8d 95 dc fe ff ff    	lea    -0x124(%ebp),%edx
  401b6a:	59                   	pop    %ecx
  401b6b:	e8 2c 0e 00 00       	call   0x40299c
  401b70:	6a 5d                	push   $0x5d
  401b72:	8d 95 a4 fe ff ff    	lea    -0x15c(%ebp),%edx
  401b78:	59                   	pop    %ecx
  401b79:	e8 1e 0e 00 00       	call   0x40299c
  401b7e:	6a 5e                	push   $0x5e
  401b80:	8d 95 10 ff ff ff    	lea    -0xf0(%ebp),%edx
  401b86:	59                   	pop    %ecx
  401b87:	e8 10 0e 00 00       	call   0x40299c
  401b8c:	6a 5f                	push   $0x5f
  401b8e:	8d 95 40 ff ff ff    	lea    -0xc0(%ebp),%edx
  401b94:	59                   	pop    %ecx
  401b95:	e8 02 0e 00 00       	call   0x40299c
  401b9a:	6a 60                	push   $0x60
  401b9c:	8d 95 70 ff ff ff    	lea    -0x90(%ebp),%edx
  401ba2:	59                   	pop    %ecx
  401ba3:	e8 f4 0d 00 00       	call   0x40299c
  401ba8:	6a 62                	push   $0x62
  401baa:	8d 55 00             	lea    0x0(%ebp),%edx
  401bad:	59                   	pop    %ecx
  401bae:	e8 e9 0d 00 00       	call   0x40299c
  401bb3:	6a 63                	push   $0x63
  401bb5:	8d 55 a0             	lea    -0x60(%ebp),%edx
  401bb8:	59                   	pop    %ecx
  401bb9:	e8 de 0d 00 00       	call   0x40299c
  401bbe:	6a 64                	push   $0x64
  401bc0:	8d 95 48 fd ff ff    	lea    -0x2b8(%ebp),%edx
  401bc6:	59                   	pop    %ecx
  401bc7:	e8 d0 0d 00 00       	call   0x40299c
  401bcc:	6a 65                	push   $0x65
  401bce:	8d 95 88 fe ff ff    	lea    -0x178(%ebp),%edx
  401bd4:	59                   	pop    %ecx
  401bd5:	e8 c2 0d 00 00       	call   0x40299c
  401bda:	6a 66                	push   $0x66
  401bdc:	8d 95 58 ff ff ff    	lea    -0xa8(%ebp),%edx
  401be2:	59                   	pop    %ecx
  401be3:	e8 b4 0d 00 00       	call   0x40299c
  401be8:	6a 67                	push   $0x67
  401bea:	8d 95 04 fe ff ff    	lea    -0x1fc(%ebp),%edx
  401bf0:	59                   	pop    %ecx
  401bf1:	e8 a6 0d 00 00       	call   0x40299c
  401bf6:	6a 68                	push   $0x68
  401bf8:	8d 95 6c fe ff ff    	lea    -0x194(%ebp),%edx
  401bfe:	59                   	pop    %ecx
  401bff:	e8 98 0d 00 00       	call   0x40299c
  401c04:	6a 69                	push   $0x69
  401c06:	8d 95 28 ff ff ff    	lea    -0xd8(%ebp),%edx
  401c0c:	59                   	pop    %ecx
  401c0d:	e8 8a 0d 00 00       	call   0x40299c
  401c12:	6a 6a                	push   $0x6a
  401c14:	8d 55 e8             	lea    -0x18(%ebp),%edx
  401c17:	59                   	pop    %ecx
  401c18:	e8 7f 0d 00 00       	call   0x40299c
  401c1d:	6a 6b                	push   $0x6b
  401c1f:	8d 95 28 fe ff ff    	lea    -0x1d8(%ebp),%edx
  401c25:	59                   	pop    %ecx
  401c26:	e8 71 0d 00 00       	call   0x40299c
  401c2b:	6a 6c                	push   $0x6c
  401c2d:	8d 95 bc fd ff ff    	lea    -0x244(%ebp),%edx
  401c33:	59                   	pop    %ecx
  401c34:	e8 63 0d 00 00       	call   0x40299c
  401c39:	6a 6d                	push   $0x6d
  401c3b:	8d 95 4c fe ff ff    	lea    -0x1b4(%ebp),%edx
  401c41:	59                   	pop    %ecx
  401c42:	e8 55 0d 00 00       	call   0x40299c
  401c47:	6a 6e                	push   $0x6e
  401c49:	8d 95 c0 fe ff ff    	lea    -0x140(%ebp),%edx
  401c4f:	59                   	pop    %ecx
  401c50:	e8 47 0d 00 00       	call   0x40299c
  401c55:	6a 6f                	push   $0x6f
  401c57:	8d 55 b8             	lea    -0x48(%ebp),%edx
  401c5a:	59                   	pop    %ecx
  401c5b:	e8 3c 0d 00 00       	call   0x40299c
  401c60:	6a 70                	push   $0x70
  401c62:	8d 95 74 fd ff ff    	lea    -0x28c(%ebp),%edx
  401c68:	59                   	pop    %ecx
  401c69:	e8 2e 0d 00 00       	call   0x40299c
  401c6e:	6a 71                	push   $0x71
  401c70:	8d 95 e0 fd ff ff    	lea    -0x220(%ebp),%edx
  401c76:	59                   	pop    %ecx
  401c77:	e8 20 0d 00 00       	call   0x40299c
  401c7c:	6a 61                	push   $0x61
  401c7e:	8d 55 d0             	lea    -0x30(%ebp),%edx
  401c81:	59                   	pop    %ecx
  401c82:	e8 15 0d 00 00       	call   0x40299c
  401c87:	8d 85 f8 fe ff ff    	lea    -0x108(%ebp),%eax
  401c8d:	33 f6                	xor    %esi,%esi
  401c8f:	89 45 14             	mov    %eax,0x14(%ebp)
  401c92:	8d 85 98 fd ff ff    	lea    -0x268(%ebp),%eax
  401c98:	89 45 18             	mov    %eax,0x18(%ebp)
  401c9b:	8d 45 88             	lea    -0x78(%ebp),%eax
  401c9e:	89 45 1c             	mov    %eax,0x1c(%ebp)
  401ca1:	8d 85 dc fe ff ff    	lea    -0x124(%ebp),%eax
  401ca7:	89 45 20             	mov    %eax,0x20(%ebp)
  401caa:	8d 85 a4 fe ff ff    	lea    -0x15c(%ebp),%eax
  401cb0:	89 45 24             	mov    %eax,0x24(%ebp)
  401cb3:	8d 85 10 ff ff ff    	lea    -0xf0(%ebp),%eax
  401cb9:	89 45 28             	mov    %eax,0x28(%ebp)
  401cbc:	8d 85 40 ff ff ff    	lea    -0xc0(%ebp),%eax
  401cc2:	89 45 2c             	mov    %eax,0x2c(%ebp)
  401cc5:	8d 85 70 ff ff ff    	lea    -0x90(%ebp),%eax
  401ccb:	89 45 30             	mov    %eax,0x30(%ebp)
  401cce:	8d 45 00             	lea    0x0(%ebp),%eax
  401cd1:	89 45 34             	mov    %eax,0x34(%ebp)
  401cd4:	8d 45 a0             	lea    -0x60(%ebp),%eax
  401cd7:	89 45 38             	mov    %eax,0x38(%ebp)
  401cda:	8d 85 48 fd ff ff    	lea    -0x2b8(%ebp),%eax
  401ce0:	89 45 3c             	mov    %eax,0x3c(%ebp)
  401ce3:	8d 85 88 fe ff ff    	lea    -0x178(%ebp),%eax
  401ce9:	89 45 40             	mov    %eax,0x40(%ebp)
  401cec:	8d 85 58 ff ff ff    	lea    -0xa8(%ebp),%eax
  401cf2:	89 45 44             	mov    %eax,0x44(%ebp)
  401cf5:	8d 85 04 fe ff ff    	lea    -0x1fc(%ebp),%eax
  401cfb:	89 45 48             	mov    %eax,0x48(%ebp)
  401cfe:	8d 85 6c fe ff ff    	lea    -0x194(%ebp),%eax
  401d04:	89 45 4c             	mov    %eax,0x4c(%ebp)
  401d07:	8d 85 28 ff ff ff    	lea    -0xd8(%ebp),%eax
  401d0d:	89 45 50             	mov    %eax,0x50(%ebp)
  401d10:	8d 45 e8             	lea    -0x18(%ebp),%eax
  401d13:	89 45 54             	mov    %eax,0x54(%ebp)
  401d16:	8d 85 28 fe ff ff    	lea    -0x1d8(%ebp),%eax
  401d1c:	89 45 58             	mov    %eax,0x58(%ebp)
  401d1f:	8d 85 bc fd ff ff    	lea    -0x244(%ebp),%eax
  401d25:	89 45 5c             	mov    %eax,0x5c(%ebp)
  401d28:	8d 85 4c fe ff ff    	lea    -0x1b4(%ebp),%eax
  401d2e:	89 45 60             	mov    %eax,0x60(%ebp)
  401d31:	8d 85 c0 fe ff ff    	lea    -0x140(%ebp),%eax
  401d37:	89 45 64             	mov    %eax,0x64(%ebp)
  401d3a:	8d 45 b8             	lea    -0x48(%ebp),%eax
  401d3d:	89 45 68             	mov    %eax,0x68(%ebp)
  401d40:	8d 85 74 fd ff ff    	lea    -0x28c(%ebp),%eax
  401d46:	89 45 6c             	mov    %eax,0x6c(%ebp)
  401d49:	8d 85 e0 fd ff ff    	lea    -0x220(%ebp),%eax
  401d4f:	89 45 70             	mov    %eax,0x70(%ebp)
  401d52:	8d 45 d0             	lea    -0x30(%ebp),%eax
  401d55:	89 45 74             	mov    %eax,0x74(%ebp)
  401d58:	8b 4c b5 14          	mov    0x14(%ebp,%esi,4),%ecx
  401d5c:	e8 ae fc ff ff       	call   0x401a0f
  401d61:	85 c0                	test   %eax,%eax
  401d63:	75 08                	jne    0x401d6d
  401d65:	46                   	inc    %esi
  401d66:	83 fe 19             	cmp    $0x19,%esi
  401d69:	72 ed                	jb     0x401d58
  401d6b:	eb 03                	jmp    0x401d70
  401d6d:	33 c0                	xor    %eax,%eax
  401d6f:	40                   	inc    %eax
  401d70:	5e                   	pop    %esi
  401d71:	8d 65 78             	lea    0x78(%ebp),%esp
  401d74:	5d                   	pop    %ebp
  401d75:	c3                   	ret
  401d76:	55                   	push   %ebp
  401d77:	8b ec                	mov    %esp,%ebp
  401d79:	83 ec 20             	sub    $0x20,%esp
  401d7c:	8d 55 f4             	lea    -0xc(%ebp),%edx
  401d7f:	6a 4f                	push   $0x4f
  401d81:	59                   	pop    %ecx
  401d82:	e8 cd 0b 00 00       	call   0x402954
  401d87:	6a 08                	push   $0x8
  401d89:	8d 55 e0             	lea    -0x20(%ebp),%edx
  401d8c:	59                   	pop    %ecx
  401d8d:	e8 0a 0c 00 00       	call   0x40299c
  401d92:	8d 45 f4             	lea    -0xc(%ebp),%eax
  401d95:	50                   	push   %eax
  401d96:	8d 45 e0             	lea    -0x20(%ebp),%eax
  401d99:	50                   	push   %eax
  401d9a:	ff 15 50 60 40 00    	call   *0x406050
  401da0:	50                   	push   %eax
  401da1:	ff 15 68 60 40 00    	call   *0x406068
  401da7:	85 c0                	test   %eax,%eax
  401da9:	74 17                	je     0x401dc2
  401dab:	8a 08                	mov    (%eax),%cl
  401dad:	80 f9 ff             	cmp    $0xff,%cl
  401db0:	75 06                	jne    0x401db8
  401db2:	80 78 01 25          	cmpb   $0x25,0x1(%eax)
  401db6:	74 05                	je     0x401dbd
  401db8:	80 f9 e9             	cmp    $0xe9,%cl
  401dbb:	75 05                	jne    0x401dc2
  401dbd:	33 c0                	xor    %eax,%eax
  401dbf:	40                   	inc    %eax
  401dc0:	eb 02                	jmp    0x401dc4
  401dc2:	33 c0                	xor    %eax,%eax
  401dc4:	8b e5                	mov    %ebp,%esp
  401dc6:	5d                   	pop    %ebp
  401dc7:	c3                   	ret
  401dc8:	55                   	push   %ebp
  401dc9:	8b ec                	mov    %esp,%ebp
  401dcb:	83 ec 30             	sub    $0x30,%esp
  401dce:	8d 55 d0             	lea    -0x30(%ebp),%edx
  401dd1:	6a 0c                	push   $0xc
  401dd3:	59                   	pop    %ecx
  401dd4:	e8 7b 0b 00 00       	call   0x402954
  401dd9:	6a 08                	push   $0x8
  401ddb:	8d 55 ec             	lea    -0x14(%ebp),%edx
  401dde:	59                   	pop    %ecx
  401ddf:	e8 b8 0b 00 00       	call   0x40299c
  401de4:	8d 45 d0             	lea    -0x30(%ebp),%eax
  401de7:	50                   	push   %eax
  401de8:	8d 45 ec             	lea    -0x14(%ebp),%eax
  401deb:	50                   	push   %eax
  401dec:	ff 15 50 60 40 00    	call   *0x406050
  401df2:	50                   	push   %eax
  401df3:	ff 15 68 60 40 00    	call   *0x406068
  401df9:	85 c0                	test   %eax,%eax
  401dfb:	74 17                	je     0x401e14
  401dfd:	8a 08                	mov    (%eax),%cl
  401dff:	80 f9 ff             	cmp    $0xff,%cl
  401e02:	75 06                	jne    0x401e0a
  401e04:	80 78 01 25          	cmpb   $0x25,0x1(%eax)
  401e08:	74 05                	je     0x401e0f
  401e0a:	80 f9 e9             	cmp    $0xe9,%cl
  401e0d:	75 05                	jne    0x401e14
  401e0f:	33 c0                	xor    %eax,%eax
  401e11:	40                   	inc    %eax
  401e12:	eb 02                	jmp    0x401e16
  401e14:	33 c0                	xor    %eax,%eax
  401e16:	8b e5                	mov    %ebp,%esp
  401e18:	5d                   	pop    %ebp
  401e19:	c3                   	ret
  401e1a:	55                   	push   %ebp
  401e1b:	8b ec                	mov    %esp,%ebp
  401e1d:	81 ec 20 02 00 00    	sub    $0x220,%esp
  401e23:	b8 08 02 00 00       	mov    $0x208,%eax
  401e28:	48                   	dec    %eax
  401e29:	c6 84 05 e0 fd ff ff 	movb   $0x0,-0x220(%ebp,%eax,1)
  401e30:	00 
  401e31:	75 f5                	jne    0x401e28
  401e33:	68 04 01 00 00       	push   $0x104
  401e38:	8d 85 e0 fd ff ff    	lea    -0x220(%ebp),%eax
  401e3e:	50                   	push   %eax
  401e3f:	6a 00                	push   $0x0
  401e41:	ff 15 6c 60 40 00    	call   *0x40606c
  401e47:	6a 36                	push   $0x36
  401e49:	8d 55 e8             	lea    -0x18(%ebp),%edx
  401e4c:	59                   	pop    %ecx
  401e4d:	e8 4a 0b 00 00       	call   0x40299c
  401e52:	51                   	push   %ecx
  401e53:	51                   	push   %ecx
  401e54:	8d 55 e8             	lea    -0x18(%ebp),%edx
  401e57:	8d 8d e0 fd ff ff    	lea    -0x220(%ebp),%ecx
  401e5d:	e8 93 09 00 00       	call   0x4027f5
  401e62:	f7 d8                	neg    %eax
  401e64:	1b c0                	sbb    %eax,%eax
  401e66:	40                   	inc    %eax
  401e67:	8b e5                	mov    %ebp,%esp
  401e69:	5d                   	pop    %ebp
  401e6a:	c3                   	ret
  401e6b:	55                   	push   %ebp
  401e6c:	8b ec                	mov    %esp,%ebp
  401e6e:	81 ec 24 02 00 00    	sub    $0x224,%esp
  401e74:	b8 08 02 00 00       	mov    $0x208,%eax
  401e79:	48                   	dec    %eax
  401e7a:	c6 84 05 dc fd ff ff 	movb   $0x0,-0x224(%ebp,%eax,1)
  401e81:	00 
  401e82:	75 f5                	jne    0x401e79
  401e84:	68 04 01 00 00       	push   $0x104
  401e89:	8d 85 dc fd ff ff    	lea    -0x224(%ebp),%eax
  401e8f:	50                   	push   %eax
  401e90:	6a 00                	push   $0x0
  401e92:	ff 15 6c 60 40 00    	call   *0x40606c
  401e98:	6a 37                	push   $0x37
  401e9a:	8d 55 e4             	lea    -0x1c(%ebp),%edx
  401e9d:	59                   	pop    %ecx
  401e9e:	e8 f9 0a 00 00       	call   0x40299c
  401ea3:	8d 55 e4             	lea    -0x1c(%ebp),%edx
  401ea6:	8d 8d dc fd ff ff    	lea    -0x224(%ebp),%ecx
  401eac:	e8 f6 09 00 00       	call   0x4028a7
  401eb1:	f7 d8                	neg    %eax
  401eb3:	1b c0                	sbb    %eax,%eax
  401eb5:	f7 d8                	neg    %eax
  401eb7:	8b e5                	mov    %ebp,%esp
  401eb9:	5d                   	pop    %ebp
  401eba:	c3                   	ret
  401ebb:	55                   	push   %ebp
  401ebc:	8b ec                	mov    %esp,%ebp
  401ebe:	83 ec 50             	sub    $0x50,%esp
  401ec1:	8d 55 cc             	lea    -0x34(%ebp),%edx
  401ec4:	56                   	push   %esi
  401ec5:	57                   	push   %edi
  401ec6:	6a 09                	push   $0x9
  401ec8:	59                   	pop    %ecx
  401ec9:	e8 ce 0a 00 00       	call   0x40299c
  401ece:	6a 10                	push   $0x10
  401ed0:	8d 55 e8             	lea    -0x18(%ebp),%edx
  401ed3:	59                   	pop    %ecx
  401ed4:	e8 7b 0a 00 00       	call   0x402954
  401ed9:	6a 11                	push   $0x11
  401edb:	8d 55 b0             	lea    -0x50(%ebp),%edx
  401ede:	59                   	pop    %ecx
  401edf:	e8 70 0a 00 00       	call   0x402954
  401ee4:	8d 45 e8             	lea    -0x18(%ebp),%eax
  401ee7:	50                   	push   %eax
  401ee8:	8d 45 cc             	lea    -0x34(%ebp),%eax
  401eeb:	50                   	push   %eax
  401eec:	ff 15 50 60 40 00    	call   *0x406050
  401ef2:	8b 35 68 60 40 00    	mov    0x406068,%esi
  401ef8:	50                   	push   %eax
  401ef9:	ff d6                	call   *%esi
  401efb:	8b f8                	mov    %eax,%edi
  401efd:	8d 45 b0             	lea    -0x50(%ebp),%eax
  401f00:	50                   	push   %eax
  401f01:	8d 45 cc             	lea    -0x34(%ebp),%eax
  401f04:	50                   	push   %eax
  401f05:	ff 15 50 60 40 00    	call   *0x406050
  401f0b:	50                   	push   %eax
  401f0c:	ff d6                	call   *%esi
  401f0e:	8b f0                	mov    %eax,%esi
  401f10:	ff d7                	call   *%edi
  401f12:	85 c0                	test   %eax,%eax
  401f14:	74 05                	je     0x401f1b
  401f16:	33 c0                	xor    %eax,%eax
  401f18:	40                   	inc    %eax
  401f19:	eb 14                	jmp    0x401f2f
  401f1b:	83 65 fc 00          	andl   $0x0,-0x4(%ebp)
  401f1f:	8d 45 fc             	lea    -0x4(%ebp),%eax
  401f22:	50                   	push   %eax
  401f23:	6a ff                	push   $0xffffffff
  401f25:	ff d6                	call   *%esi
  401f27:	33 c0                	xor    %eax,%eax
  401f29:	39 45 fc             	cmp    %eax,-0x4(%ebp)
  401f2c:	0f 95 c0             	setne  %al
  401f2f:	5f                   	pop    %edi
  401f30:	5e                   	pop    %esi
  401f31:	8b e5                	mov    %ebp,%esp
  401f33:	5d                   	pop    %ebp
  401f34:	c3                   	ret
  401f35:	55                   	push   %ebp
  401f36:	8b ec                	mov    %esp,%ebp
  401f38:	81 ec 74 04 00 00    	sub    $0x474,%esp
  401f3e:	56                   	push   %esi
  401f3f:	8d 45 fc             	lea    -0x4(%ebp),%eax
  401f42:	c7 45 fc 12 04 00 00 	movl   $0x412,-0x4(%ebp)
  401f49:	50                   	push   %eax
  401f4a:	8d 85 8c fb ff ff    	lea    -0x474(%ebp),%eax
  401f50:	50                   	push   %eax
  401f51:	ff 15 04 60 40 00    	call   *0x406004
  401f57:	8d 8d 8c fb ff ff    	lea    -0x474(%ebp),%ecx
  401f5d:	33 f6                	xor    %esi,%esi
  401f5f:	e8 7c 08 00 00       	call   0x4027e0
  401f64:	85 c0                	test   %eax,%eax
  401f66:	7e 25                	jle    0x401f8d
  401f68:	66 8b 8c 75 8c fb ff 	mov    -0x474(%ebp,%esi,2),%cx
  401f6f:	ff 
  401f70:	e8 8f 09 00 00       	call   0x402904
  401f75:	66 89 84 75 8c fb ff 	mov    %ax,-0x474(%ebp,%esi,2)
  401f7c:	ff 
  401f7d:	8d 8d 8c fb ff ff    	lea    -0x474(%ebp),%ecx
  401f83:	46                   	inc    %esi
  401f84:	e8 57 08 00 00       	call   0x4027e0
  401f89:	3b f0                	cmp    %eax,%esi
  401f8b:	7c db                	jl     0x401f68
  401f8d:	6a 39                	push   $0x39
  401f8f:	8d 55 c0             	lea    -0x40(%ebp),%edx
  401f92:	59                   	pop    %ecx
  401f93:	e8 04 0a 00 00       	call   0x40299c
  401f98:	6a 3a                	push   $0x3a
  401f9a:	8d 55 f0             	lea    -0x10(%ebp),%edx
  401f9d:	59                   	pop    %ecx
  401f9e:	e8 f9 09 00 00       	call   0x40299c
  401fa3:	6a 3b                	push   $0x3b
  401fa5:	8d 55 e0             	lea    -0x20(%ebp),%edx
  401fa8:	59                   	pop    %ecx
  401fa9:	e8 ee 09 00 00       	call   0x40299c
  401fae:	6a 3c                	push   $0x3c
  401fb0:	8d 55 d0             	lea    -0x30(%ebp),%edx
  401fb3:	59                   	pop    %ecx
  401fb4:	e8 e3 09 00 00       	call   0x40299c
  401fb9:	6a 3d                	push   $0x3d
  401fbb:	8d 55 a0             	lea    -0x60(%ebp),%edx
  401fbe:	59                   	pop    %ecx
  401fbf:	e8 d8 09 00 00       	call   0x40299c
  401fc4:	8d 55 c0             	lea    -0x40(%ebp),%edx
  401fc7:	8d 8d 8c fb ff ff    	lea    -0x474(%ebp),%ecx
  401fcd:	e8 d5 08 00 00       	call   0x4028a7
  401fd2:	5e                   	pop    %esi
  401fd3:	85 c0                	test   %eax,%eax
  401fd5:	75 48                	jne    0x40201f
  401fd7:	8d 55 f0             	lea    -0x10(%ebp),%edx
  401fda:	8d 8d 8c fb ff ff    	lea    -0x474(%ebp),%ecx
  401fe0:	e8 c2 08 00 00       	call   0x4028a7
  401fe5:	85 c0                	test   %eax,%eax
  401fe7:	75 36                	jne    0x40201f
  401fe9:	8d 55 e0             	lea    -0x20(%ebp),%edx
  401fec:	8d 8d 8c fb ff ff    	lea    -0x474(%ebp),%ecx
  401ff2:	e8 b0 08 00 00       	call   0x4028a7
  401ff7:	85 c0                	test   %eax,%eax
  401ff9:	75 24                	jne    0x40201f
  401ffb:	8d 55 d0             	lea    -0x30(%ebp),%edx
  401ffe:	8d 8d 8c fb ff ff    	lea    -0x474(%ebp),%ecx
  402004:	e8 9e 08 00 00       	call   0x4028a7
  402009:	85 c0                	test   %eax,%eax
  40200b:	75 12                	jne    0x40201f
  40200d:	8d 55 a0             	lea    -0x60(%ebp),%edx
  402010:	8d 8d 8c fb ff ff    	lea    -0x474(%ebp),%ecx
  402016:	e8 8c 08 00 00       	call   0x4028a7
  40201b:	85 c0                	test   %eax,%eax
  40201d:	74 03                	je     0x402022
  40201f:	33 c0                	xor    %eax,%eax
  402021:	40                   	inc    %eax
  402022:	8b e5                	mov    %ebp,%esp
  402024:	5d                   	pop    %ebp
  402025:	c3                   	ret
  402026:	55                   	push   %ebp
  402027:	8b ec                	mov    %esp,%ebp
  402029:	81 ec 34 02 00 00    	sub    $0x234,%esp
  40202f:	8d 85 cc fd ff ff    	lea    -0x234(%ebp),%eax
  402035:	56                   	push   %esi
  402036:	68 08 02 00 00       	push   $0x208
  40203b:	50                   	push   %eax
  40203c:	33 f6                	xor    %esi,%esi
  40203e:	56                   	push   %esi
  40203f:	ff 15 6c 60 40 00    	call   *0x40606c
  402045:	8d 8d cc fd ff ff    	lea    -0x234(%ebp),%ecx
  40204b:	e8 90 07 00 00       	call   0x4027e0
  402050:	85 c0                	test   %eax,%eax
  402052:	7e 25                	jle    0x402079
  402054:	66 8b 8c 75 cc fd ff 	mov    -0x234(%ebp,%esi,2),%cx
  40205b:	ff 
  40205c:	e8 a3 08 00 00       	call   0x402904
  402061:	66 89 84 75 cc fd ff 	mov    %ax,-0x234(%ebp,%esi,2)
  402068:	ff 
  402069:	8d 8d cc fd ff ff    	lea    -0x234(%ebp),%ecx
  40206f:	46                   	inc    %esi
  402070:	e8 6b 07 00 00       	call   0x4027e0
  402075:	3b f0                	cmp    %eax,%esi
  402077:	7c db                	jl     0x402054
  402079:	6a 3e                	push   $0x3e
  40207b:	8d 55 e4             	lea    -0x1c(%ebp),%edx
  40207e:	59                   	pop    %ecx
  40207f:	e8 18 09 00 00       	call   0x40299c
  402084:	6a 3a                	push   $0x3a
  402086:	8d 55 f4             	lea    -0xc(%ebp),%edx
  402089:	59                   	pop    %ecx
  40208a:	e8 0d 09 00 00       	call   0x40299c
  40208f:	6a 39                	push   $0x39
  402091:	8d 55 d4             	lea    -0x2c(%ebp),%edx
  402094:	59                   	pop    %ecx
  402095:	e8 02 09 00 00       	call   0x40299c
  40209a:	8d 55 e4             	lea    -0x1c(%ebp),%edx
  40209d:	8d 8d cc fd ff ff    	lea    -0x234(%ebp),%ecx
  4020a3:	e8 ff 07 00 00       	call   0x4028a7
  4020a8:	5e                   	pop    %esi
  4020a9:	85 c0                	test   %eax,%eax
  4020ab:	75 24                	jne    0x4020d1
  4020ad:	8d 55 f4             	lea    -0xc(%ebp),%edx
  4020b0:	8d 8d cc fd ff ff    	lea    -0x234(%ebp),%ecx
  4020b6:	e8 ec 07 00 00       	call   0x4028a7
  4020bb:	85 c0                	test   %eax,%eax
  4020bd:	75 12                	jne    0x4020d1
  4020bf:	8d 55 d4             	lea    -0x2c(%ebp),%edx
  4020c2:	8d 8d cc fd ff ff    	lea    -0x234(%ebp),%ecx
  4020c8:	e8 da 07 00 00       	call   0x4028a7
  4020cd:	85 c0                	test   %eax,%eax
  4020cf:	74 03                	je     0x4020d4
  4020d1:	33 c0                	xor    %eax,%eax
  4020d3:	40                   	inc    %eax
  4020d4:	8b e5                	mov    %ebp,%esp
  4020d6:	5d                   	pop    %ebp
  4020d7:	c3                   	ret
  4020d8:	64 a1 30 00 00 00    	mov    %fs:0x30,%eax
  4020de:	56                   	push   %esi
  4020df:	57                   	push   %edi
  4020e0:	8b 78 18             	mov    0x18(%eax),%edi
  4020e3:	e8 92 f6 ff ff       	call   0x40177a
  4020e8:	8b f0                	mov    %eax,%esi
  4020ea:	e8 8b f6 ff ff       	call   0x40177a
  4020ef:	83 fe 04             	cmp    $0x4,%esi
  4020f2:	1b c9                	sbb    %ecx,%ecx
  4020f4:	83 e1 cc             	and    $0xffffffcc,%ecx
  4020f7:	f7 44 39 40 fd ff ff 	testl  $0xfffffffd,0x40(%ecx,%edi,1)
  4020fe:	ff 
  4020ff:	75 13                	jne    0x402114
  402101:	83 f8 04             	cmp    $0x4,%eax
  402104:	1b c0                	sbb    %eax,%eax
  402106:	83 e0 cc             	and    $0xffffffcc,%eax
  402109:	83 7c 38 44 00       	cmpl   $0x0,0x44(%eax,%edi,1)
  40210e:	75 04                	jne    0x402114
  402110:	33 c0                	xor    %eax,%eax
  402112:	eb 03                	jmp    0x402117
  402114:	33 c0                	xor    %eax,%eax
  402116:	40                   	inc    %eax
  402117:	5f                   	pop    %edi
  402118:	5e                   	pop    %esi
  402119:	c3                   	ret
  40211a:	55                   	push   %ebp
  40211b:	8b ec                	mov    %esp,%ebp
  40211d:	83 ec 6c             	sub    $0x6c,%esp
  402120:	8d 55 e0             	lea    -0x20(%ebp),%edx
  402123:	53                   	push   %ebx
  402124:	56                   	push   %esi
  402125:	6a 08                	push   $0x8
  402127:	59                   	pop    %ecx
  402128:	e8 6f 08 00 00       	call   0x40299c
  40212d:	6a 0c                	push   $0xc
  40212f:	8d 55 94             	lea    -0x6c(%ebp),%edx
  402132:	59                   	pop    %ecx
  402133:	e8 1c 08 00 00       	call   0x402954
  402138:	8d 45 94             	lea    -0x6c(%ebp),%eax
  40213b:	50                   	push   %eax
  40213c:	8d 45 e0             	lea    -0x20(%ebp),%eax
  40213f:	50                   	push   %eax
  402140:	ff 15 50 60 40 00    	call   *0x406050
  402146:	50                   	push   %eax
  402147:	ff 15 68 60 40 00    	call   *0x406068
  40214d:	8b f0                	mov    %eax,%esi
  40214f:	85 f6                	test   %esi,%esi
  402151:	0f 84 86 00 00 00    	je     0x4021dd
  402157:	33 db                	xor    %ebx,%ebx
  402159:	8d 45 fc             	lea    -0x4(%ebp),%eax
  40215c:	53                   	push   %ebx
  40215d:	6a 04                	push   $0x4
  40215f:	50                   	push   %eax
  402160:	6a 07                	push   $0x7
  402162:	6a ff                	push   $0xffffffff
  402164:	89 5d fc             	mov    %ebx,-0x4(%ebp)
  402167:	ff d6                	call   *%esi
  402169:	85 c0                	test   %eax,%eax
  40216b:	75 0a                	jne    0x402177
  40216d:	39 5d fc             	cmp    %ebx,-0x4(%ebp)
  402170:	74 05                	je     0x402177
  402172:	33 c0                	xor    %eax,%eax
  402174:	40                   	inc    %eax
  402175:	eb 68                	jmp    0x4021df
  402177:	53                   	push   %ebx
  402178:	6a 04                	push   $0x4
  40217a:	8d 45 f8             	lea    -0x8(%ebp),%eax
  40217d:	89 5d f8             	mov    %ebx,-0x8(%ebp)
  402180:	50                   	push   %eax
  402181:	6a 1e                	push   $0x1e
  402183:	6a ff                	push   $0xffffffff
  402185:	ff d6                	call   *%esi
  402187:	85 c0                	test   %eax,%eax
  402189:	75 05                	jne    0x402190
  40218b:	39 5d f8             	cmp    %ebx,-0x8(%ebp)
  40218e:	75 e2                	jne    0x402172
  402190:	53                   	push   %ebx
  402191:	6a 04                	push   $0x4
  402193:	8d 45 f4             	lea    -0xc(%ebp),%eax
  402196:	89 5d f4             	mov    %ebx,-0xc(%ebp)
  402199:	50                   	push   %eax
  40219a:	6a 1f                	push   $0x1f
  40219c:	6a ff                	push   $0xffffffff
  40219e:	ff d6                	call   *%esi
  4021a0:	85 c0                	test   %eax,%eax
  4021a2:	75 05                	jne    0x4021a9
  4021a4:	39 5d f4             	cmp    %ebx,-0xc(%ebp)
  4021a7:	74 c9                	je     0x402172
  4021a9:	6a 18                	push   $0x18
  4021ab:	58                   	pop    %eax
  4021ac:	48                   	dec    %eax
  4021ad:	88 5c 05 b0          	mov    %bl,-0x50(%ebp,%eax,1)
  4021b1:	75 f9                	jne    0x4021ac
  4021b3:	53                   	push   %ebx
  4021b4:	6a 18                	push   $0x18
  4021b6:	8d 45 b0             	lea    -0x50(%ebp),%eax
  4021b9:	50                   	push   %eax
  4021ba:	53                   	push   %ebx
  4021bb:	6a ff                	push   $0xffffffff
  4021bd:	ff d6                	call   *%esi
  4021bf:	6a 50                	push   $0x50
  4021c1:	8d 55 c8             	lea    -0x38(%ebp),%edx
  4021c4:	59                   	pop    %ecx
  4021c5:	e8 d2 07 00 00       	call   0x40299c
  4021ca:	8b 4d c4             	mov    -0x3c(%ebp),%ecx
  4021cd:	85 c9                	test   %ecx,%ecx
  4021cf:	74 0c                	je     0x4021dd
  4021d1:	8d 55 c8             	lea    -0x38(%ebp),%edx
  4021d4:	e8 bf f7 ff ff       	call   0x401998
  4021d9:	85 c0                	test   %eax,%eax
  4021db:	75 95                	jne    0x402172
  4021dd:	33 c0                	xor    %eax,%eax
  4021df:	5e                   	pop    %esi
  4021e0:	5b                   	pop    %ebx
  4021e1:	8b e5                	mov    %ebp,%esp
  4021e3:	5d                   	pop    %ebp
  4021e4:	c3                   	ret
  4021e5:	55                   	push   %ebp
  4021e6:	8b ec                	mov    %esp,%ebp
  4021e8:	81 ec cc 02 00 00    	sub    $0x2cc,%esp
  4021ee:	b8 cc 02 00 00       	mov    $0x2cc,%eax
  4021f3:	48                   	dec    %eax
  4021f4:	c6 84 05 34 fd ff ff 	movb   $0x0,-0x2cc(%ebp,%eax,1)
  4021fb:	00 
  4021fc:	75 f5                	jne    0x4021f3
  4021fe:	8d 85 34 fd ff ff    	lea    -0x2cc(%ebp),%eax
  402204:	c7 85 34 fd ff ff 10 	movl   $0x10010,-0x2cc(%ebp)
  40220b:	00 01 00 
  40220e:	50                   	push   %eax
  40220f:	ff 15 94 60 40 00    	call   *0x406094
  402215:	50                   	push   %eax
  402216:	ff 15 28 60 40 00    	call   *0x406028
  40221c:	85 c0                	test   %eax,%eax
  40221e:	74 29                	je     0x402249
  402220:	83 bd 38 fd ff ff 00 	cmpl   $0x0,-0x2c8(%ebp)
  402227:	75 1b                	jne    0x402244
  402229:	83 bd 3c fd ff ff 00 	cmpl   $0x0,-0x2c4(%ebp)
  402230:	75 12                	jne    0x402244
  402232:	83 bd 40 fd ff ff 00 	cmpl   $0x0,-0x2c0(%ebp)
  402239:	75 09                	jne    0x402244
  40223b:	83 bd 44 fd ff ff 00 	cmpl   $0x0,-0x2bc(%ebp)
  402242:	74 05                	je     0x402249
  402244:	33 c0                	xor    %eax,%eax
  402246:	40                   	inc    %eax
  402247:	eb 02                	jmp    0x40224b
  402249:	33 c0                	xor    %eax,%eax
  40224b:	8b e5                	mov    %ebp,%esp
  40224d:	5d                   	pop    %ebp
  40224e:	c3                   	ret
  40224f:	55                   	push   %ebp
  402250:	8b ec                	mov    %esp,%ebp
  402252:	81 ec 40 02 00 00    	sub    $0x240,%esp
  402258:	53                   	push   %ebx
  402259:	56                   	push   %esi
  40225a:	6a 51                	push   $0x51
  40225c:	8d 55 ec             	lea    -0x14(%ebp),%edx
  40225f:	59                   	pop    %ecx
  402260:	e8 37 07 00 00       	call   0x40299c
  402265:	33 db                	xor    %ebx,%ebx
  402267:	53                   	push   %ebx
  402268:	6a 02                	push   $0x2
  40226a:	ff 15 24 60 40 00    	call   *0x406024
  402270:	8b f0                	mov    %eax,%esi
  402272:	c7 85 c0 fd ff ff 2c 	movl   $0x22c,-0x240(%ebp)
  402279:	02 00 00 
  40227c:	8d 85 c0 fd ff ff    	lea    -0x240(%ebp),%eax
  402282:	50                   	push   %eax
  402283:	56                   	push   %esi
  402284:	ff 15 3c 60 40 00    	call   *0x40603c
  40228a:	85 c0                	test   %eax,%eax
  40228c:	74 30                	je     0x4022be
  40228e:	eb 14                	jmp    0x4022a4
  402290:	51                   	push   %ecx
  402291:	51                   	push   %ecx
  402292:	8d 95 e4 fd ff ff    	lea    -0x21c(%ebp),%edx
  402298:	8d 4d ec             	lea    -0x14(%ebp),%ecx
  40229b:	e8 55 05 00 00       	call   0x4027f5
  4022a0:	85 c0                	test   %eax,%eax
  4022a2:	74 14                	je     0x4022b8
  4022a4:	8d 85 c0 fd ff ff    	lea    -0x240(%ebp),%eax
  4022aa:	50                   	push   %eax
  4022ab:	56                   	push   %esi
  4022ac:	ff 15 20 60 40 00    	call   *0x406020
  4022b2:	85 c0                	test   %eax,%eax
  4022b4:	75 da                	jne    0x402290
  4022b6:	eb 06                	jmp    0x4022be
  4022b8:	8b 9d c8 fd ff ff    	mov    -0x238(%ebp),%ebx
  4022be:	56                   	push   %esi
  4022bf:	8b 35 80 60 40 00    	mov    0x406080,%esi
  4022c5:	ff d6                	call   *%esi
  4022c7:	53                   	push   %ebx
  4022c8:	6a 00                	push   $0x0
  4022ca:	68 ff ff 1f 00       	push   $0x1fffff
  4022cf:	ff 15 44 60 40 00    	call   *0x406044
  4022d5:	85 c0                	test   %eax,%eax
  4022d7:	74 08                	je     0x4022e1
  4022d9:	50                   	push   %eax
  4022da:	ff d6                	call   *%esi
  4022dc:	33 c0                	xor    %eax,%eax
  4022de:	40                   	inc    %eax
  4022df:	eb 02                	jmp    0x4022e3
  4022e1:	33 c0                	xor    %eax,%eax
  4022e3:	5e                   	pop    %esi
  4022e4:	5b                   	pop    %ebx
  4022e5:	8b e5                	mov    %ebp,%esp
  4022e7:	5d                   	pop    %ebp
  4022e8:	c3                   	ret
  4022e9:	56                   	push   %esi
  4022ea:	8b 74 24 08          	mov    0x8(%esp),%esi
  4022ee:	8b 46 04             	mov    0x4(%esi),%eax
  4022f1:	ff b0 b0 00 00 00    	push   0xb0(%eax)
  4022f7:	ff 15 30 60 40 00    	call   *0x406030
  4022fd:	8b 46 04             	mov    0x4(%esi),%eax
  402300:	5e                   	pop    %esi
  402301:	83 80 b8 00 00 00 02 	addl   $0x2,0xb8(%eax)
  402308:	83 c8 ff             	or     $0xffffffff,%eax
  40230b:	c2 04 00             	ret    $0x4
  40230e:	6a 24                	push   $0x24
  402310:	68 b0 72 40 00       	push   $0x4072b0
  402315:	e8 96 08 00 00       	call   0x402bb0
  40231a:	8d 55 cc             	lea    -0x34(%ebp),%edx
  40231d:	6a 08                	push   $0x8
  40231f:	59                   	pop    %ecx
  402320:	e8 77 06 00 00       	call   0x40299c
  402325:	8d 55 e0             	lea    -0x20(%ebp),%edx
  402328:	6a 0f                	push   $0xf
  40232a:	59                   	pop    %ecx
  40232b:	e8 24 06 00 00       	call   0x402954
  402330:	8d 45 e0             	lea    -0x20(%ebp),%eax
  402333:	50                   	push   %eax
  402334:	8d 45 cc             	lea    -0x34(%ebp),%eax
  402337:	50                   	push   %eax
  402338:	ff 15 50 60 40 00    	call   *0x406050
  40233e:	50                   	push   %eax
  40233f:	ff 15 68 60 40 00    	call   *0x406068
  402345:	85 c0                	test   %eax,%eax
  402347:	74 0f                	je     0x402358
  402349:	83 65 fc 00          	andl   $0x0,-0x4(%ebp)
  40234d:	68 99 99 99 99       	push   $0x99999999
  402352:	ff d0                	call   *%eax
  402354:	83 4d fc ff          	orl    $0xffffffff,-0x4(%ebp)
  402358:	33 c0                	xor    %eax,%eax
  40235a:	e8 8c 08 00 00       	call   0x402beb
  40235f:	c3                   	ret
  402360:	33 c0                	xor    %eax,%eax
  402362:	40                   	inc    %eax
  402363:	c3                   	ret
  402364:	8b 65 e8             	mov    -0x18(%ebp),%esp
  402367:	83 4d fc ff          	orl    $0xffffffff,-0x4(%ebp)
  40236b:	33 c0                	xor    %eax,%eax
  40236d:	40                   	inc    %eax
  40236e:	eb ea                	jmp    0x40235a
  402370:	8b 4c 24 04          	mov    0x4(%esp),%ecx
  402374:	83 25 f8 a4 41 00 00 	andl   $0x0,0x41a4f8
  40237b:	8b 01                	mov    (%ecx),%eax
  40237d:	81 38 03 00 00 80    	cmpl   $0x80000003,(%eax)
  402383:	75 0e                	jne    0x402393
  402385:	8b 41 04             	mov    0x4(%ecx),%eax
  402388:	ff 80 b8 00 00 00    	incl   0xb8(%eax)
  40238e:	83 c8 ff             	or     $0xffffffff,%eax
  402391:	eb 02                	jmp    0x402395
  402393:	33 c0                	xor    %eax,%eax
  402395:	c2 04 00             	ret    $0x4
  402398:	6a 08                	push   $0x8
  40239a:	68 c0 72 40 00       	push   $0x4072c0
  40239f:	e8 0c 08 00 00       	call   0x402bb0
  4023a4:	83 65 fc 00          	andl   $0x0,-0x4(%ebp)
  4023a8:	68 99 99 99 99       	push   $0x99999999
  4023ad:	ff 15 80 60 40 00    	call   *0x406080
  4023b3:	83 4d fc ff          	orl    $0xffffffff,-0x4(%ebp)
  4023b7:	33 c0                	xor    %eax,%eax
  4023b9:	eb 0e                	jmp    0x4023c9
  4023bb:	33 c0                	xor    %eax,%eax
  4023bd:	40                   	inc    %eax
  4023be:	c3                   	ret
  4023bf:	8b 65 e8             	mov    -0x18(%ebp),%esp
  4023c2:	83 4d fc ff          	orl    $0xffffffff,-0x4(%ebp)
  4023c6:	33 c0                	xor    %eax,%eax
  4023c8:	40                   	inc    %eax
  4023c9:	e8 1d 08 00 00       	call   0x402beb
  4023ce:	c3                   	ret
  4023cf:	81 ec 38 01 00 00    	sub    $0x138,%esp
  4023d5:	8d 54 24 1c          	lea    0x1c(%esp),%edx
  4023d9:	55                   	push   %ebp
  4023da:	56                   	push   %esi
  4023db:	57                   	push   %edi
  4023dc:	6a 07                	push   $0x7
  4023de:	59                   	pop    %ecx
  4023df:	e8 b8 05 00 00       	call   0x40299c
  4023e4:	6a 0e                	push   $0xe
  4023e6:	8d 54 24 10          	lea    0x10(%esp),%edx
  4023ea:	59                   	pop    %ecx
  4023eb:	e8 64 05 00 00       	call   0x402954
  4023f0:	8d 44 24 28          	lea    0x28(%esp),%eax
  4023f4:	50                   	push   %eax
  4023f5:	ff 15 34 60 40 00    	call   *0x406034
  4023fb:	33 ed                	xor    %ebp,%ebp
  4023fd:	45                   	inc    %ebp
  4023fe:	85 c0                	test   %eax,%eax
  402400:	74 13                	je     0x402415
  402402:	8d 4c 24 0c          	lea    0xc(%esp),%ecx
  402406:	51                   	push   %ecx
  402407:	50                   	push   %eax
  402408:	ff 15 68 60 40 00    	call   *0x406068
  40240e:	85 c0                	test   %eax,%eax
  402410:	74 03                	je     0x402415
  402412:	55                   	push   %ebp
  402413:	ff d0                	call   *%eax
  402415:	6a 2c                	push   $0x2c
  402417:	8d 54 24 74          	lea    0x74(%esp),%edx
  40241b:	59                   	pop    %ecx
  40241c:	e8 7b 05 00 00       	call   0x40299c
  402421:	6a 2d                	push   $0x2d
  402423:	8d 94 24 90 00 00 00 	lea    0x90(%esp),%edx
  40242a:	59                   	pop    %ecx
  40242b:	e8 6c 05 00 00       	call   0x40299c
  402430:	8b 3d 50 60 40 00    	mov    0x406050,%edi
  402436:	8d 44 24 70          	lea    0x70(%esp),%eax
  40243a:	50                   	push   %eax
  40243b:	ff d7                	call   *%edi
  40243d:	85 c0                	test   %eax,%eax
  40243f:	0f 85 b5 01 00 00    	jne    0x4025fa
  402445:	8d 84 24 8c 00 00 00 	lea    0x8c(%esp),%eax
  40244c:	50                   	push   %eax
  40244d:	ff d7                	call   *%edi
  40244f:	85 c0                	test   %eax,%eax
  402451:	0f 85 a3 01 00 00    	jne    0x4025fa
  402457:	6a 53                	push   $0x53
  402459:	8d 54 24 1c          	lea    0x1c(%esp),%edx
  40245d:	59                   	pop    %ecx
  40245e:	e8 39 05 00 00       	call   0x40299c
  402463:	8b 35 b0 60 40 00    	mov    0x4060b0,%esi
  402469:	8d 44 24 18          	lea    0x18(%esp),%eax
  40246d:	6a 00                	push   $0x0
  40246f:	50                   	push   %eax
  402470:	ff d6                	call   *%esi
  402472:	85 c0                	test   %eax,%eax
  402474:	0f 85 80 01 00 00    	jne    0x4025fa
  40247a:	6a 54                	push   $0x54
  40247c:	8d 94 24 00 01 00 00 	lea    0x100(%esp),%edx
  402483:	59                   	pop    %ecx
  402484:	e8 13 05 00 00       	call   0x40299c
  402489:	6a 00                	push   $0x0
  40248b:	8d 84 24 00 01 00 00 	lea    0x100(%esp),%eax
  402492:	50                   	push   %eax
  402493:	ff d6                	call   *%esi
  402495:	85 c0                	test   %eax,%eax
  402497:	0f 85 5d 01 00 00    	jne    0x4025fa
  40249d:	6a 55                	push   $0x55
  40249f:	8d 94 24 24 01 00 00 	lea    0x124(%esp),%edx
  4024a6:	59                   	pop    %ecx
  4024a7:	e8 f0 04 00 00       	call   0x40299c
  4024ac:	6a 00                	push   $0x0
  4024ae:	8d 84 24 24 01 00 00 	lea    0x124(%esp),%eax
  4024b5:	50                   	push   %eax
  4024b6:	ff d6                	call   *%esi
  4024b8:	85 c0                	test   %eax,%eax
  4024ba:	0f 85 3a 01 00 00    	jne    0x4025fa
  4024c0:	6a 56                	push   $0x56
  4024c2:	8d 94 24 c8 00 00 00 	lea    0xc8(%esp),%edx
  4024c9:	59                   	pop    %ecx
  4024ca:	e8 cd 04 00 00       	call   0x40299c
  4024cf:	6a 00                	push   $0x0
  4024d1:	8d 84 24 c8 00 00 00 	lea    0xc8(%esp),%eax
  4024d8:	50                   	push   %eax
  4024d9:	ff d6                	call   *%esi
  4024db:	85 c0                	test   %eax,%eax
  4024dd:	0f 85 17 01 00 00    	jne    0x4025fa
  4024e3:	6a 57                	push   $0x57
  4024e5:	8d 94 24 e4 00 00 00 	lea    0xe4(%esp),%edx
  4024ec:	59                   	pop    %ecx
  4024ed:	e8 aa 04 00 00       	call   0x40299c
  4024f2:	6a 00                	push   $0x0
  4024f4:	8d 84 24 e4 00 00 00 	lea    0xe4(%esp),%eax
  4024fb:	50                   	push   %eax
  4024fc:	ff d6                	call   *%esi
  4024fe:	85 c0                	test   %eax,%eax
  402500:	0f 85 f4 00 00 00    	jne    0x4025fa
  402506:	6a 58                	push   $0x58
  402508:	8d 54 24 44          	lea    0x44(%esp),%edx
  40250c:	59                   	pop    %ecx
  40250d:	e8 8a 04 00 00       	call   0x40299c
  402512:	6a 00                	push   $0x0
  402514:	8d 44 24 44          	lea    0x44(%esp),%eax
  402518:	50                   	push   %eax
  402519:	ff d6                	call   *%esi
  40251b:	85 c0                	test   %eax,%eax
  40251d:	0f 85 d7 00 00 00    	jne    0x4025fa
  402523:	8a 0d d4 02 fe 7f    	mov    0x7ffe02d4,%cl
  402529:	f6 c1 03             	test   $0x3,%cl
  40252c:	0f 85 c8 00 00 00    	jne    0x4025fa
  402532:	e8 84 f9 ff ff       	call   0x401ebb
  402537:	85 c0                	test   %eax,%eax
  402539:	0f 85 bb 00 00 00    	jne    0x4025fa
  40253f:	e8 36 f5 ff ff       	call   0x401a7a
  402544:	85 c0                	test   %eax,%eax
  402546:	0f 85 ae 00 00 00    	jne    0x4025fa
  40254c:	e8 de f5 ff ff       	call   0x401b2f
  402551:	85 c0                	test   %eax,%eax
  402553:	0f 85 a1 00 00 00    	jne    0x4025fa
  402559:	6a 09                	push   $0x9
  40255b:	8d 94 24 ac 00 00 00 	lea    0xac(%esp),%edx
  402562:	59                   	pop    %ecx
  402563:	e8 34 04 00 00       	call   0x40299c
  402568:	6a 38                	push   $0x38
  40256a:	8d 54 24 5c          	lea    0x5c(%esp),%edx
  40256e:	59                   	pop    %ecx
  40256f:	e8 e0 03 00 00       	call   0x402954
  402574:	8d 44 24 58          	lea    0x58(%esp),%eax
  402578:	50                   	push   %eax
  402579:	8d 84 24 ac 00 00 00 	lea    0xac(%esp),%eax
  402580:	50                   	push   %eax
  402581:	ff d7                	call   *%edi
  402583:	50                   	push   %eax
  402584:	ff 15 68 60 40 00    	call   *0x406068
  40258a:	85 c0                	test   %eax,%eax
  40258c:	75 6c                	jne    0x4025fa
  40258e:	e8 7b fd ff ff       	call   0x40230e
  402593:	85 c0                	test   %eax,%eax
  402595:	75 63                	jne    0x4025fa
  402597:	e8 fc fd ff ff       	call   0x402398
  40259c:	85 c0                	test   %eax,%eax
  40259e:	75 5a                	jne    0x4025fa
  4025a0:	68 70 23 40 00       	push   $0x402370
  4025a5:	55                   	push   %ebp
  4025a6:	ff 15 38 60 40 00    	call   *0x406038
  4025ac:	89 2d f8 a4 41 00    	mov    %ebp,0x41a4f8
  4025b2:	cc                   	int3
  4025b3:	50                   	push   %eax
  4025b4:	ff 15 2c 60 40 00    	call   *0x40602c
  4025ba:	83 3d f8 a4 41 00 00 	cmpl   $0x0,0x41a4f8
  4025c1:	75 37                	jne    0x4025fa
  4025c3:	8b 0d 48 60 40 00    	mov    0x406048,%ecx
  4025c9:	8a 01                	mov    (%ecx),%al
  4025cb:	3c ff                	cmp    $0xff,%al
  4025cd:	75 06                	jne    0x4025d5
  4025cf:	80 79 01 25          	cmpb   $0x25,0x1(%ecx)
  4025d3:	74 25                	je     0x4025fa
  4025d5:	3c e9                	cmp    $0xe9,%al
  4025d7:	74 21                	je     0x4025fa
  4025d9:	a1 7c 60 40 00       	mov    0x40607c,%eax
  4025de:	8a 08                	mov    (%eax),%cl
  4025e0:	80 f9 ff             	cmp    $0xff,%cl
  4025e3:	75 21                	jne    0x402606
  4025e5:	80 78 01 25          	cmpb   $0x25,0x1(%eax)
  4025e9:	75 1b                	jne    0x402606
  4025eb:	8b 40 02             	mov    0x2(%eax),%eax
  4025ee:	8b 00                	mov    (%eax),%eax
  4025f0:	80 38 8b             	cmpb   $0x8b,(%eax)
  4025f3:	75 05                	jne    0x4025fa
  4025f5:	38 48 01             	cmp    %cl,0x1(%eax)
  4025f8:	74 11                	je     0x40260b
  4025fa:	8b c5                	mov    %ebp,%eax
  4025fc:	5f                   	pop    %edi
  4025fd:	5e                   	pop    %esi
  4025fe:	5d                   	pop    %ebp
  4025ff:	81 c4 38 01 00 00    	add    $0x138,%esp
  402605:	c3                   	ret
  402606:	80 f9 e9             	cmp    $0xe9,%cl
  402609:	74 ef                	je     0x4025fa
  40260b:	e8 66 f7 ff ff       	call   0x401d76
  402610:	85 c0                	test   %eax,%eax
  402612:	75 e6                	jne    0x4025fa
  402614:	e8 af f7 ff ff       	call   0x401dc8
  402619:	85 c0                	test   %eax,%eax
  40261b:	75 dd                	jne    0x4025fa
  40261d:	e8 2d fc ff ff       	call   0x40224f
  402622:	85 c0                	test   %eax,%eax
  402624:	75 d4                	jne    0x4025fa
  402626:	68 e9 22 40 00       	push   $0x4022e9
  40262b:	ff 15 30 60 40 00    	call   *0x406030
  402631:	33 c0                	xor    %eax,%eax
  402633:	f7 f0                	div    %eax
  402635:	64 a1 30 00 00 00    	mov    %fs:0x30,%eax
  40263b:	f6 40 68 70          	testb  $0x70,0x68(%eax)
  40263f:	75 b9                	jne    0x4025fa
  402641:	e8 92 fa ff ff       	call   0x4020d8
  402646:	85 c0                	test   %eax,%eax
  402648:	75 b0                	jne    0x4025fa
  40264a:	e8 cb fa ff ff       	call   0x40211a
  40264f:	85 c0                	test   %eax,%eax
  402651:	75 a7                	jne    0x4025fa
  402653:	e8 8d fb ff ff       	call   0x4021e5
  402658:	85 c0                	test   %eax,%eax
  40265a:	75 9e                	jne    0x4025fa
  40265c:	e8 b9 f7 ff ff       	call   0x401e1a
  402661:	85 c0                	test   %eax,%eax
  402663:	75 95                	jne    0x4025fa
  402665:	e8 01 f8 ff ff       	call   0x401e6b
  40266a:	85 c0                	test   %eax,%eax
  40266c:	75 8c                	jne    0x4025fa
  40266e:	e8 c2 f8 ff ff       	call   0x401f35
  402673:	85 c0                	test   %eax,%eax
  402675:	75 83                	jne    0x4025fa
  402677:	e8 aa f9 ff ff       	call   0x402026
  40267c:	f7 d8                	neg    %eax
  40267e:	1b c0                	sbb    %eax,%eax
  402680:	f7 d8                	neg    %eax
  402682:	e9 75 ff ff ff       	jmp    0x4025fc
  402687:	55                   	push   %ebp
  402688:	8b ec                	mov    %esp,%ebp
  40268a:	83 ec 10             	sub    $0x10,%esp
  40268d:	56                   	push   %esi
  40268e:	8b f1                	mov    %ecx,%esi
  402690:	8d 4d f0             	lea    -0x10(%ebp),%ecx
  402693:	8b 46 28             	mov    0x28(%esi),%eax
  402696:	8d 56 44             	lea    0x44(%esi),%edx
  402699:	83 e8 44             	sub    $0x44,%eax
  40269c:	50                   	push   %eax
  40269d:	e8 5e e9 ff ff       	call   0x401000
  4026a2:	51                   	push   %ecx
  4026a3:	8d 56 34             	lea    0x34(%esi),%edx
  4026a6:	8d 4d f0             	lea    -0x10(%ebp),%ecx
  4026a9:	e8 a3 f0 ff ff       	call   0x401751
  4026ae:	f7 d8                	neg    %eax
  4026b0:	5e                   	pop    %esi
  4026b1:	1b c0                	sbb    %eax,%eax
  4026b3:	40                   	inc    %eax
  4026b4:	8b e5                	mov    %ebp,%esp
  4026b6:	5d                   	pop    %ebp
  4026b7:	c3                   	ret
  4026b8:	85 d2                	test   %edx,%edx
  4026ba:	75 13                	jne    0x4026cf
  4026bc:	39 51 30             	cmp    %edx,0x30(%ecx)
  4026bf:	74 0b                	je     0x4026cc
  4026c1:	83 79 28 54          	cmpl   $0x54,0x28(%ecx)
  4026c5:	72 05                	jb     0x4026cc
  4026c7:	8d 41 44             	lea    0x44(%ecx),%eax
  4026ca:	eb 0b                	jmp    0x4026d7
  4026cc:	33 c0                	xor    %eax,%eax
  4026ce:	c3                   	ret
  4026cf:	8b 42 08             	mov    0x8(%edx),%eax
  4026d2:	83 c0 10             	add    $0x10,%eax
  4026d5:	03 c2                	add    %edx,%eax
  4026d7:	8b d0                	mov    %eax,%edx
  4026d9:	2b d1                	sub    %ecx,%edx
  4026db:	56                   	push   %esi
  4026dc:	8b 71 28             	mov    0x28(%ecx),%esi
  4026df:	83 c2 10             	add    $0x10,%edx
  4026e2:	3b d6                	cmp    %esi,%edx
  4026e4:	77 09                	ja     0x4026ef
  4026e6:	8b 48 08             	mov    0x8(%eax),%ecx
  4026e9:	03 ca                	add    %edx,%ecx
  4026eb:	3b ce                	cmp    %esi,%ecx
  4026ed:	76 02                	jbe    0x4026f1
  4026ef:	33 c0                	xor    %eax,%eax
  4026f1:	5e                   	pop    %esi
  4026f2:	c3                   	ret
  4026f3:	56                   	push   %esi
  4026f4:	8b f1                	mov    %ecx,%esi
  4026f6:	33 d2                	xor    %edx,%edx
  4026f8:	eb 12                	jmp    0x40270c
  4026fa:	83 38 0b             	cmpl   $0xb,(%eax)
  4026fd:	75 09                	jne    0x402708
  4026ff:	f7 40 04 00 00 00 70 	testl  $0x70000000,0x4(%eax)
  402706:	74 0d                	je     0x402715
  402708:	8b d0                	mov    %eax,%edx
  40270a:	8b ce                	mov    %esi,%ecx
  40270c:	e8 a7 ff ff ff       	call   0x4026b8
  402711:	85 c0                	test   %eax,%eax
  402713:	75 e5                	jne    0x4026fa
  402715:	5e                   	pop    %esi
  402716:	c2 04 00             	ret    $0x4
  402719:	51                   	push   %ecx
  40271a:	56                   	push   %esi
  40271b:	57                   	push   %edi
  40271c:	8b f9                	mov    %ecx,%edi
  40271e:	33 f6                	xor    %esi,%esi
  402720:	8b 4f 0c             	mov    0xc(%edi),%ecx
  402723:	85 c9                	test   %ecx,%ecx
  402725:	74 41                	je     0x402768
  402727:	83 c1 02             	add    $0x2,%ecx
  40272a:	e8 e1 ef ff ff       	call   0x401710
  40272f:	8b f0                	mov    %eax,%esi
  402731:	85 f6                	test   %esi,%esi
  402733:	74 33                	je     0x402768
  402735:	f6 47 04 01          	testb  $0x1,0x4(%edi)
  402739:	74 20                	je     0x40275b
  40273b:	8b 57 08             	mov    0x8(%edi),%edx
  40273e:	8d 4c 24 08          	lea    0x8(%esp),%ecx
  402742:	51                   	push   %ecx
  402743:	56                   	push   %esi
  402744:	8d 4f 10             	lea    0x10(%edi),%ecx
  402747:	e8 ac 02 00 00       	call   0x4029f8
  40274c:	85 c0                	test   %eax,%eax
  40274e:	75 18                	jne    0x402768
  402750:	8b ce                	mov    %esi,%ecx
  402752:	e8 d0 ef ff ff       	call   0x401727
  402757:	33 f6                	xor    %esi,%esi
  402759:	eb 0d                	jmp    0x402768
  40275b:	ff 77 0c             	push   0xc(%edi)
  40275e:	8d 57 10             	lea    0x10(%edi),%edx
  402761:	8b ce                	mov    %esi,%ecx
  402763:	e8 cf ef ff ff       	call   0x401737
  402768:	5f                   	pop    %edi
  402769:	8b c6                	mov    %esi,%eax
  40276b:	5e                   	pop    %esi
  40276c:	59                   	pop    %ecx
  40276d:	c3                   	ret
  40276e:	53                   	push   %ebx
  40276f:	8b 5c 24 08          	mov    0x8(%esp),%ebx
  402773:	56                   	push   %esi
  402774:	57                   	push   %edi
  402775:	8b fa                	mov    %edx,%edi
  402777:	8b f1                	mov    %ecx,%esi
  402779:	8d 43 bc             	lea    -0x44(%ebx),%eax
  40277c:	3d bc ff 9f 00       	cmp    $0x9fffbc,%eax
  402781:	77 4e                	ja     0x4027d1
  402783:	85 f6                	test   %esi,%esi
  402785:	75 15                	jne    0x40279c
  402787:	39 5f 28             	cmp    %ebx,0x28(%edi)
  40278a:	77 4c                	ja     0x4027d8
  40278c:	8b cf                	mov    %edi,%ecx
  40278e:	e8 f4 fe ff ff       	call   0x402687
  402793:	85 c0                	test   %eax,%eax
  402795:	74 41                	je     0x4027d8
  402797:	8b 47 28             	mov    0x28(%edi),%eax
  40279a:	eb 3e                	jmp    0x4027da
  40279c:	8b cb                	mov    %ebx,%ecx
  40279e:	e8 6d ef ff ff       	call   0x401710
  4027a3:	89 06                	mov    %eax,(%esi)
  4027a5:	85 c0                	test   %eax,%eax
  4027a7:	74 28                	je     0x4027d1
  4027a9:	53                   	push   %ebx
  4027aa:	8b d7                	mov    %edi,%edx
  4027ac:	8b c8                	mov    %eax,%ecx
  4027ae:	e8 84 ef ff ff       	call   0x401737
  4027b3:	8b 0e                	mov    (%esi),%ecx
  4027b5:	39 59 28             	cmp    %ebx,0x28(%ecx)
  4027b8:	77 10                	ja     0x4027ca
  4027ba:	e8 c8 fe ff ff       	call   0x402687
  4027bf:	85 c0                	test   %eax,%eax
  4027c1:	74 07                	je     0x4027ca
  4027c3:	8b 06                	mov    (%esi),%eax
  4027c5:	8b 40 28             	mov    0x28(%eax),%eax
  4027c8:	eb 10                	jmp    0x4027da
  4027ca:	8b 0e                	mov    (%esi),%ecx
  4027cc:	e8 56 ef ff ff       	call   0x401727
  4027d1:	85 f6                	test   %esi,%esi
  4027d3:	74 03                	je     0x4027d8
  4027d5:	83 26 00             	andl   $0x0,(%esi)
  4027d8:	33 c0                	xor    %eax,%eax
  4027da:	5f                   	pop    %edi
  4027db:	5e                   	pop    %esi
  4027dc:	5b                   	pop    %ebx
  4027dd:	c2 08 00             	ret    $0x8
  4027e0:	33 d2                	xor    %edx,%edx
  4027e2:	8b c2                	mov    %edx,%eax
  4027e4:	85 c9                	test   %ecx,%ecx
  4027e6:	74 0c                	je     0x4027f4
  4027e8:	66 39 11             	cmp    %dx,(%ecx)
  4027eb:	74 07                	je     0x4027f4
  4027ed:	40                   	inc    %eax
  4027ee:	66 39 14 41          	cmp    %dx,(%ecx,%eax,2)
  4027f2:	75 f9                	jne    0x4027ed
  4027f4:	c3                   	ret
  4027f5:	53                   	push   %ebx
  4027f6:	55                   	push   %ebp
  4027f7:	8b e9                	mov    %ecx,%ebp
  4027f9:	8b da                	mov    %edx,%ebx
  4027fb:	85 ed                	test   %ebp,%ebp
  4027fd:	75 0d                	jne    0x40280c
  4027ff:	85 db                	test   %ebx,%ebx
  402801:	74 05                	je     0x402808
  402803:	83 c8 ff             	or     $0xffffffff,%eax
  402806:	eb 69                	jmp    0x402871
  402808:	33 c0                	xor    %eax,%eax
  40280a:	eb 65                	jmp    0x402871
  40280c:	85 db                	test   %ebx,%ebx
  40280e:	75 05                	jne    0x402815
  402810:	33 c0                	xor    %eax,%eax
  402812:	40                   	inc    %eax
  402813:	eb 5c                	jmp    0x402871
  402815:	0f b7 0b             	movzwl (%ebx),%ecx
  402818:	56                   	push   %esi
  402819:	57                   	push   %edi
  40281a:	e8 e5 00 00 00       	call   0x402904
  40281f:	66 8b 4d 00          	mov    0x0(%ebp),%cx
  402823:	0f b7 f0             	movzwl %ax,%esi
  402826:	e8 d9 00 00 00       	call   0x402904
  40282b:	0f b7 f8             	movzwl %ax,%edi
  40282e:	2b fe                	sub    %esi,%edi
  402830:	75 2a                	jne    0x40285c
  402832:	0f b7 33             	movzwl (%ebx),%esi
  402835:	2b eb                	sub    %ebx,%ebp
  402837:	66 85 f6             	test   %si,%si
  40283a:	74 20                	je     0x40285c
  40283c:	83 c3 02             	add    $0x2,%ebx
  40283f:	66 8b 0c 2b          	mov    (%ebx,%ebp,1),%cx
  402843:	0f b7 33             	movzwl (%ebx),%esi
  402846:	e8 b9 00 00 00       	call   0x402904
  40284b:	8b ce                	mov    %esi,%ecx
  40284d:	0f b7 f8             	movzwl %ax,%edi
  402850:	e8 af 00 00 00       	call   0x402904
  402855:	0f b7 c0             	movzwl %ax,%eax
  402858:	2b f8                	sub    %eax,%edi
  40285a:	74 db                	je     0x402837
  40285c:	33 c0                	xor    %eax,%eax
  40285e:	83 c9 ff             	or     $0xffffffff,%ecx
  402861:	40                   	inc    %eax
  402862:	85 ff                	test   %edi,%edi
  402864:	0f 4f c8             	cmovg  %eax,%ecx
  402867:	f7 df                	neg    %edi
  402869:	1b ff                	sbb    %edi,%edi
  40286b:	23 f9                	and    %ecx,%edi
  40286d:	8b c7                	mov    %edi,%eax
  40286f:	5f                   	pop    %edi
  402870:	5e                   	pop    %esi
  402871:	5d                   	pop    %ebp
  402872:	5b                   	pop    %ebx
  402873:	c2 08 00             	ret    $0x8
  402876:	51                   	push   %ecx
  402877:	56                   	push   %esi
  402878:	8b f1                	mov    %ecx,%esi
  40287a:	85 f6                	test   %esi,%esi
  40287c:	75 04                	jne    0x402882
  40287e:	33 c0                	xor    %eax,%eax
  402880:	eb 22                	jmp    0x4028a4
  402882:	57                   	push   %edi
  402883:	e8 58 ff ff ff       	call   0x4027e0
  402888:	8d 3c 00             	lea    (%eax,%eax,1),%edi
  40288b:	8d 4f 02             	lea    0x2(%edi),%ecx
  40288e:	e8 7d ee ff ff       	call   0x401710
  402893:	8b c8                	mov    %eax,%ecx
  402895:	85 c9                	test   %ecx,%ecx
  402897:	74 08                	je     0x4028a1
  402899:	57                   	push   %edi
  40289a:	8b d6                	mov    %esi,%edx
  40289c:	e8 96 ee ff ff       	call   0x401737
  4028a1:	8b c1                	mov    %ecx,%eax
  4028a3:	5f                   	pop    %edi
  4028a4:	5e                   	pop    %esi
  4028a5:	59                   	pop    %ecx
  4028a6:	c3                   	ret
  4028a7:	53                   	push   %ebx
  4028a8:	8b da                	mov    %edx,%ebx
  4028aa:	55                   	push   %ebp
  4028ab:	33 ed                	xor    %ebp,%ebp
  4028ad:	56                   	push   %esi
  4028ae:	8b f1                	mov    %ecx,%esi
  4028b0:	66 39 2b             	cmp    %bp,(%ebx)
  4028b3:	75 04                	jne    0x4028b9
  4028b5:	8b c6                	mov    %esi,%eax
  4028b7:	eb 43                	jmp    0x4028fc
  4028b9:	0f b7 06             	movzwl (%esi),%eax
  4028bc:	57                   	push   %edi
  4028bd:	66 85 c0             	test   %ax,%ax
  4028c0:	74 37                	je     0x4028f9
  4028c2:	8b fe                	mov    %esi,%edi
  4028c4:	2b fb                	sub    %ebx,%edi
  4028c6:	8b d3                	mov    %ebx,%edx
  4028c8:	66 85 c0             	test   %ax,%ax
  4028cb:	74 19                	je     0x4028e6
  4028cd:	0f b7 02             	movzwl (%edx),%eax
  4028d0:	66 85 c0             	test   %ax,%ax
  4028d3:	74 2b                	je     0x402900
  4028d5:	0f b7 0c 17          	movzwl (%edi,%edx,1),%ecx
  4028d9:	2b c8                	sub    %eax,%ecx
  4028db:	75 09                	jne    0x4028e6
  4028dd:	83 c2 02             	add    $0x2,%edx
  4028e0:	66 39 2c 17          	cmp    %bp,(%edi,%edx,1)
  4028e4:	75 e7                	jne    0x4028cd
  4028e6:	66 39 2a             	cmp    %bp,(%edx)
  4028e9:	74 15                	je     0x402900
  4028eb:	83 c6 02             	add    $0x2,%esi
  4028ee:	83 c7 02             	add    $0x2,%edi
  4028f1:	0f b7 06             	movzwl (%esi),%eax
  4028f4:	66 85 c0             	test   %ax,%ax
  4028f7:	75 cd                	jne    0x4028c6
  4028f9:	33 c0                	xor    %eax,%eax
  4028fb:	5f                   	pop    %edi
  4028fc:	5e                   	pop    %esi
  4028fd:	5d                   	pop    %ebp
  4028fe:	5b                   	pop    %ebx
  4028ff:	c3                   	ret
  402900:	8b c6                	mov    %esi,%eax
  402902:	eb f7                	jmp    0x4028fb
  402904:	b8 00 02 00 00       	mov    $0x200,%eax
  402909:	66 3b c8             	cmp    %ax,%cx
  40290c:	73 0d                	jae    0x40291b
  40290e:	0f b7 c9             	movzwl %cx,%ecx
  402911:	0f be 81 78 63 40 00 	movsbl 0x406378(%ecx),%eax
  402918:	03 c1                	add    %ecx,%eax
  40291a:	c3                   	ret
  40291b:	56                   	push   %esi
  40291c:	be 80 65 40 00       	mov    $0x406580,%esi
  402921:	b8 a0 03 00 00       	mov    $0x3a0,%eax
  402926:	66 3b c8             	cmp    %ax,%cx
  402929:	72 11                	jb     0x40293c
  40292b:	66 3b 4e 02          	cmp    0x2(%esi),%cx
  40292f:	76 10                	jbe    0x402941
  402931:	83 c6 08             	add    $0x8,%esi
  402934:	0f b7 06             	movzwl (%esi),%eax
  402937:	66 85 c0             	test   %ax,%ax
  40293a:	75 ea                	jne    0x402926
  40293c:	66 8b c1             	mov    %cx,%ax
  40293f:	5e                   	pop    %esi
  402940:	c3                   	ret
  402941:	0f b7 06             	movzwl (%esi),%eax
  402944:	0f b7 d1             	movzwl %cx,%edx
  402947:	8b 4e 04             	mov    0x4(%esi),%ecx
  40294a:	2b c8                	sub    %eax,%ecx
  40294c:	5e                   	pop    %esi
  40294d:	0f be 04 11          	movsbl (%ecx,%edx,1),%eax
  402951:	03 c2                	add    %edx,%eax
  402953:	c3                   	ret
  402954:	56                   	push   %esi
  402955:	0f b7 f1             	movzwl %cx,%esi
  402958:	33 c0                	xor    %eax,%eax
  40295a:	57                   	push   %edi
  40295b:	8b fa                	mov    %edx,%edi
  40295d:	33 d2                	xor    %edx,%edx
  40295f:	66 3b 04 f5 b2 65 40 	cmp    0x4065b2(,%esi,8),%ax
  402966:	00 
  402967:	73 24                	jae    0x40298d
  402969:	8b 04 f5 b4 65 40 00 	mov    0x4065b4(,%esi,8),%eax
  402970:	0f b7 ca             	movzwl %dx,%ecx
  402973:	8a 04 08             	mov    (%eax,%ecx,1),%al
  402976:	32 04 f5 b0 65 40 00 	xor    0x4065b0(,%esi,8),%al
  40297d:	32 c2                	xor    %dl,%al
  40297f:	42                   	inc    %edx
  402980:	88 04 39             	mov    %al,(%ecx,%edi,1)
  402983:	66 3b 14 f5 b2 65 40 	cmp    0x4065b2(,%esi,8),%dx
  40298a:	00 
  40298b:	72 dc                	jb     0x402969
  40298d:	0f b7 04 f5 b2 65 40 	movzwl 0x4065b2(,%esi,8),%eax
  402994:	00 
  402995:	c6 04 38 00          	movb   $0x0,(%eax,%edi,1)
  402999:	5f                   	pop    %edi
  40299a:	5e                   	pop    %esi
  40299b:	c3                   	ret
  40299c:	53                   	push   %ebx
  40299d:	56                   	push   %esi
  40299e:	0f b7 f1             	movzwl %cx,%esi
  4029a1:	33 c0                	xor    %eax,%eax
  4029a3:	57                   	push   %edi
  4029a4:	33 ff                	xor    %edi,%edi
  4029a6:	8b da                	mov    %edx,%ebx
  4029a8:	66 3b 04 f5 b2 65 40 	cmp    0x4065b2(,%esi,8),%ax
  4029af:	00 
  4029b0:	73 34                	jae    0x4029e6
  4029b2:	8b 04 f5 b4 65 40 00 	mov    0x4065b4(,%esi,8),%eax
  4029b9:	0f b7 d7             	movzwl %di,%edx
  4029bc:	66 0f be 0c 10       	movsbw (%eax,%edx,1),%cx
  4029c1:	b8 ff 00 00 00       	mov    $0xff,%eax
  4029c6:	66 33 cf             	xor    %di,%cx
  4029c9:	66 23 c8             	and    %ax,%cx
  4029cc:	0f b6 04 f5 b0 65 40 	movzbl 0x4065b0(,%esi,8),%eax
  4029d3:	00 
  4029d4:	66 33 c8             	xor    %ax,%cx
  4029d7:	47                   	inc    %edi
  4029d8:	66 89 0c 53          	mov    %cx,(%ebx,%edx,2)
  4029dc:	66 3b 3c f5 b2 65 40 	cmp    0x4065b2(,%esi,8),%di
  4029e3:	00 
  4029e4:	72 cc                	jb     0x4029b2
  4029e6:	0f b7 04 f5 b2 65 40 	movzwl 0x4065b2(,%esi,8),%eax
  4029ed:	00 
  4029ee:	33 c9                	xor    %ecx,%ecx
  4029f0:	5f                   	pop    %edi
  4029f1:	5e                   	pop    %esi
  4029f2:	66 89 0c 43          	mov    %cx,(%ebx,%eax,2)
  4029f6:	5b                   	pop    %ebx
  4029f7:	c3                   	ret
  4029f8:	83 ec 18             	sub    $0x18,%esp
  4029fb:	53                   	push   %ebx
  4029fc:	55                   	push   %ebp
  4029fd:	56                   	push   %esi
  4029fe:	57                   	push   %edi
  4029ff:	33 ff                	xor    %edi,%edi
  402a01:	89 54 24 24          	mov    %edx,0x24(%esp)
  402a05:	33 c0                	xor    %eax,%eax
  402a07:	8b d9                	mov    %ecx,%ebx
  402a09:	40                   	inc    %eax
  402a0a:	89 5c 24 20          	mov    %ebx,0x20(%esp)
  402a0e:	8b cf                	mov    %edi,%ecx
  402a10:	89 44 24 18          	mov    %eax,0x18(%esp)
  402a14:	8b f7                	mov    %edi,%esi
  402a16:	8b ef                	mov    %edi,%ebp
  402a18:	8b 54 24 2c          	mov    0x2c(%esp),%edx
  402a1c:	85 c9                	test   %ecx,%ecx
  402a1e:	74 0a                	je     0x402a2a
  402a20:	49                   	dec    %ecx
  402a21:	8b c7                	mov    %edi,%eax
  402a23:	d3 e8                	shr    %cl,%eax
  402a25:	83 e0 01             	and    $0x1,%eax
  402a28:	eb 0e                	jmp    0x402a38
  402a2a:	8b 3c 1e             	mov    (%esi,%ebx,1),%edi
  402a2d:	83 c6 04             	add    $0x4,%esi
  402a30:	6a 1f                	push   $0x1f
  402a32:	8b c7                	mov    %edi,%eax
  402a34:	59                   	pop    %ecx
  402a35:	c1 e8 1f             	shr    $0x1f,%eax
  402a38:	85 c0                	test   %eax,%eax
  402a3a:	74 0a                	je     0x402a46
  402a3c:	8a 04 1e             	mov    (%esi,%ebx,1),%al
  402a3f:	88 04 2a             	mov    %al,(%edx,%ebp,1)
  402a42:	45                   	inc    %ebp
  402a43:	46                   	inc    %esi
  402a44:	eb d6                	jmp    0x402a1c
  402a46:	33 d2                	xor    %edx,%edx
  402a48:	89 6c 24 14          	mov    %ebp,0x14(%esp)
  402a4c:	42                   	inc    %edx
  402a4d:	85 c9                	test   %ecx,%ecx
  402a4f:	74 0a                	je     0x402a5b
  402a51:	49                   	dec    %ecx
  402a52:	8b c7                	mov    %edi,%eax
  402a54:	d3 e8                	shr    %cl,%eax
  402a56:	83 e0 01             	and    $0x1,%eax
  402a59:	eb 0e                	jmp    0x402a69
  402a5b:	8b 3c 1e             	mov    (%esi,%ebx,1),%edi
  402a5e:	83 c6 04             	add    $0x4,%esi
  402a61:	6a 1f                	push   $0x1f
  402a63:	8b c7                	mov    %edi,%eax
  402a65:	59                   	pop    %ecx
  402a66:	c1 e8 1f             	shr    $0x1f,%eax
  402a69:	8d 14 50             	lea    (%eax,%edx,2),%edx
  402a6c:	85 c9                	test   %ecx,%ecx
  402a6e:	74 0a                	je     0x402a7a
  402a70:	49                   	dec    %ecx
  402a71:	8b c7                	mov    %edi,%eax
  402a73:	d3 e8                	shr    %cl,%eax
  402a75:	83 e0 01             	and    $0x1,%eax
  402a78:	eb 0e                	jmp    0x402a88
  402a7a:	8b 3c 1e             	mov    (%esi,%ebx,1),%edi
  402a7d:	83 c6 04             	add    $0x4,%esi
  402a80:	6a 1f                	push   $0x1f
  402a82:	8b c7                	mov    %edi,%eax
  402a84:	59                   	pop    %ecx
  402a85:	c1 e8 1f             	shr    $0x1f,%eax
  402a88:	85 c0                	test   %eax,%eax
  402a8a:	74 c1                	je     0x402a4d
  402a8c:	89 74 24 10          	mov    %esi,0x10(%esp)
  402a90:	83 fa 02             	cmp    $0x2,%edx
  402a93:	75 06                	jne    0x402a9b
  402a95:	8b 6c 24 18          	mov    0x18(%esp),%ebp
  402a99:	eb 1f                	jmp    0x402aba
  402a9b:	0f b6 2c 1e          	movzbl (%esi,%ebx,1),%ebp
  402a9f:	8d 42 fd             	lea    -0x3(%edx),%eax
  402aa2:	c1 e0 08             	shl    $0x8,%eax
  402aa5:	03 e8                	add    %eax,%ebp
  402aa7:	46                   	inc    %esi
  402aa8:	89 74 24 10          	mov    %esi,0x10(%esp)
  402aac:	83 fd ff             	cmp    $0xffffffff,%ebp
  402aaf:	0f 84 dc 00 00 00    	je     0x402b91
  402ab5:	45                   	inc    %ebp
  402ab6:	89 6c 24 18          	mov    %ebp,0x18(%esp)
  402aba:	85 c9                	test   %ecx,%ecx
  402abc:	74 0a                	je     0x402ac8
  402abe:	49                   	dec    %ecx
  402abf:	8b c7                	mov    %edi,%eax
  402ac1:	d3 e8                	shr    %cl,%eax
  402ac3:	83 e0 01             	and    $0x1,%eax
  402ac6:	eb 12                	jmp    0x402ada
  402ac8:	8b 3c 1e             	mov    (%esi,%ebx,1),%edi
  402acb:	83 c6 04             	add    $0x4,%esi
  402ace:	6a 1f                	push   $0x1f
  402ad0:	8b c7                	mov    %edi,%eax
  402ad2:	89 74 24 14          	mov    %esi,0x14(%esp)
  402ad6:	59                   	pop    %ecx
  402ad7:	c1 e8 1f             	shr    $0x1f,%eax
  402ada:	85 c9                	test   %ecx,%ecx
  402adc:	74 0a                	je     0x402ae8
  402ade:	49                   	dec    %ecx
  402adf:	8b d7                	mov    %edi,%edx
  402ae1:	d3 ea                	shr    %cl,%edx
  402ae3:	83 e2 01             	and    $0x1,%edx
  402ae6:	eb 12                	jmp    0x402afa
  402ae8:	8b 3c 1e             	mov    (%esi,%ebx,1),%edi
  402aeb:	83 c6 04             	add    $0x4,%esi
  402aee:	6a 1f                	push   $0x1f
  402af0:	8b d7                	mov    %edi,%edx
  402af2:	89 74 24 14          	mov    %esi,0x14(%esp)
  402af6:	59                   	pop    %ecx
  402af7:	c1 ea 1f             	shr    $0x1f,%edx
  402afa:	8d 14 42             	lea    (%edx,%eax,2),%edx
  402afd:	85 d2                	test   %edx,%edx
  402aff:	75 47                	jne    0x402b48
  402b01:	42                   	inc    %edx
  402b02:	85 c9                	test   %ecx,%ecx
  402b04:	74 0a                	je     0x402b10
  402b06:	49                   	dec    %ecx
  402b07:	8b c7                	mov    %edi,%eax
  402b09:	d3 e8                	shr    %cl,%eax
  402b0b:	83 e0 01             	and    $0x1,%eax
  402b0e:	eb 0e                	jmp    0x402b1e
  402b10:	8b 3c 1e             	mov    (%esi,%ebx,1),%edi
  402b13:	83 c6 04             	add    $0x4,%esi
  402b16:	6a 1f                	push   $0x1f
  402b18:	8b c7                	mov    %edi,%eax
  402b1a:	59                   	pop    %ecx
  402b1b:	c1 e8 1f             	shr    $0x1f,%eax
  402b1e:	8d 14 50             	lea    (%eax,%edx,2),%edx
  402b21:	85 c9                	test   %ecx,%ecx
  402b23:	74 0a                	je     0x402b2f
  402b25:	49                   	dec    %ecx
  402b26:	8b c7                	mov    %edi,%eax
  402b28:	d3 e8                	shr    %cl,%eax
  402b2a:	83 e0 01             	and    $0x1,%eax
  402b2d:	eb 0e                	jmp    0x402b3d
  402b2f:	8b 3c 1e             	mov    (%esi,%ebx,1),%edi
  402b32:	83 c6 04             	add    $0x4,%esi
  402b35:	6a 1f                	push   $0x1f
  402b37:	8b c7                	mov    %edi,%eax
  402b39:	59                   	pop    %ecx
  402b3a:	c1 e8 1f             	shr    $0x1f,%eax
  402b3d:	85 c0                	test   %eax,%eax
  402b3f:	74 c1                	je     0x402b02
  402b41:	89 74 24 10          	mov    %esi,0x10(%esp)
  402b45:	83 c2 02             	add    $0x2,%edx
  402b48:	8b 5c 24 14          	mov    0x14(%esp),%ebx
  402b4c:	b8 00 0d 00 00       	mov    $0xd00,%eax
  402b51:	8b 74 24 2c          	mov    0x2c(%esp),%esi
  402b55:	3b c5                	cmp    %ebp,%eax
  402b57:	1b c0                	sbb    %eax,%eax
  402b59:	2b dd                	sub    %ebp,%ebx
  402b5b:	03 5c 24 2c          	add    0x2c(%esp),%ebx
  402b5f:	f7 d8                	neg    %eax
  402b61:	8b 6c 24 14          	mov    0x14(%esp),%ebp
  402b65:	03 d0                	add    %eax,%edx
  402b67:	89 54 24 1c          	mov    %edx,0x1c(%esp)
  402b6b:	8b 54 24 2c          	mov    0x2c(%esp),%edx
  402b6f:	8a 03                	mov    (%ebx),%al
  402b71:	88 04 2a             	mov    %al,(%edx,%ebp,1)
  402b74:	45                   	inc    %ebp
  402b75:	8b 54 24 1c          	mov    0x1c(%esp),%edx
  402b79:	43                   	inc    %ebx
  402b7a:	8a 03                	mov    (%ebx),%al
  402b7c:	88 04 2e             	mov    %al,(%esi,%ebp,1)
  402b7f:	45                   	inc    %ebp
  402b80:	43                   	inc    %ebx
  402b81:	4a                   	dec    %edx
  402b82:	75 f6                	jne    0x402b7a
  402b84:	8b 74 24 10          	mov    0x10(%esp),%esi
  402b88:	8b 5c 24 20          	mov    0x20(%esp),%ebx
  402b8c:	e9 87 fe ff ff       	jmp    0x402a18
  402b91:	8b 44 24 30          	mov    0x30(%esp),%eax
  402b95:	8b 5c 24 14          	mov    0x14(%esp),%ebx
  402b99:	5f                   	pop    %edi
  402b9a:	89 18                	mov    %ebx,(%eax)
  402b9c:	33 c0                	xor    %eax,%eax
  402b9e:	3b 74 24 20          	cmp    0x20(%esp),%esi
  402ba2:	5e                   	pop    %esi
  402ba3:	5d                   	pop    %ebp
  402ba4:	0f 94 c0             	sete   %al
  402ba7:	5b                   	pop    %ebx
  402ba8:	83 c4 18             	add    $0x18,%esp
  402bab:	c2 08 00             	ret    $0x8
  402bae:	cc                   	int3
  402baf:	cc                   	int3
  402bb0:	68 08 2c 40 00       	push   $0x402c08
  402bb5:	64 a1 00 00 00 00    	mov    %fs:0x0,%eax
  402bbb:	50                   	push   %eax
  402bbc:	8b 44 24 10          	mov    0x10(%esp),%eax
  402bc0:	89 6c 24 10          	mov    %ebp,0x10(%esp)
  402bc4:	8d 6c 24 10          	lea    0x10(%esp),%ebp
  402bc8:	2b e0                	sub    %eax,%esp
  402bca:	53                   	push   %ebx
  402bcb:	56                   	push   %esi
  402bcc:	57                   	push   %edi
  402bcd:	8b 45 f8             	mov    -0x8(%ebp),%eax
  402bd0:	89 65 e8             	mov    %esp,-0x18(%ebp)
  402bd3:	50                   	push   %eax
  402bd4:	8b 45 fc             	mov    -0x4(%ebp),%eax
  402bd7:	c7 45 fc ff ff ff ff 	movl   $0xffffffff,-0x4(%ebp)
  402bde:	89 45 f8             	mov    %eax,-0x8(%ebp)
  402be1:	8d 45 f0             	lea    -0x10(%ebp),%eax
  402be4:	64 a3 00 00 00 00    	mov    %eax,%fs:0x0
  402bea:	c3                   	ret
  402beb:	8b 4d f0             	mov    -0x10(%ebp),%ecx
  402bee:	64 89 0d 00 00 00 00 	mov    %ecx,%fs:0x0
  402bf5:	59                   	pop    %ecx
  402bf6:	5f                   	pop    %edi
  402bf7:	5e                   	pop    %esi
  402bf8:	5b                   	pop    %ebx
  402bf9:	c9                   	leave
  402bfa:	51                   	push   %ecx
  402bfb:	c3                   	ret
  402bfc:	cc                   	int3
  402bfd:	cc                   	int3
  402bfe:	cc                   	int3
  402bff:	cc                   	int3
  402c00:	56                   	push   %esi
  402c01:	43                   	inc    %ebx
  402c02:	32 30                	xor    (%eax),%dh
  402c04:	58                   	pop    %eax
  402c05:	43                   	inc    %ebx
  402c06:	30 30                	xor    %dh,(%eax)
  402c08:	55                   	push   %ebp
  402c09:	8b ec                	mov    %esp,%ebp
  402c0b:	83 ec 08             	sub    $0x8,%esp
  402c0e:	53                   	push   %ebx
  402c0f:	56                   	push   %esi
  402c10:	57                   	push   %edi
  402c11:	55                   	push   %ebp
  402c12:	fc                   	cld
  402c13:	8b 5d 0c             	mov    0xc(%ebp),%ebx
  402c16:	8b 45 08             	mov    0x8(%ebp),%eax
  402c19:	f7 40 04 06 00 00 00 	testl  $0x6,0x4(%eax)
  402c20:	0f 85 c3 00 00 00    	jne    0x402ce9
  402c26:	89 45 f8             	mov    %eax,-0x8(%ebp)
  402c29:	8b 45 10             	mov    0x10(%ebp),%eax
  402c2c:	89 45 fc             	mov    %eax,-0x4(%ebp)
  402c2f:	8d 45 f8             	lea    -0x8(%ebp),%eax
  402c32:	89 43 fc             	mov    %eax,-0x4(%ebx)
  402c35:	8b 73 0c             	mov    0xc(%ebx),%esi
  402c38:	8b 7b 08             	mov    0x8(%ebx),%edi
  402c3b:	53                   	push   %ebx
  402c3c:	e8 1f 03 00 00       	call   0x402f60
  402c41:	83 c4 04             	add    $0x4,%esp
  402c44:	0b c0                	or     %eax,%eax
  402c46:	0f 8e 8f 00 00 00    	jle    0x402cdb
  402c4c:	83 fe ff             	cmp    $0xffffffff,%esi
  402c4f:	0f 84 8d 00 00 00    	je     0x402ce2
  402c55:	8d 0c 76             	lea    (%esi,%esi,2),%ecx
  402c58:	8b 44 8f 04          	mov    0x4(%edi,%ecx,4),%eax
  402c5c:	0b c0                	or     %eax,%eax
  402c5e:	74 66                	je     0x402cc6
  402c60:	56                   	push   %esi
  402c61:	55                   	push   %ebp
  402c62:	8d 6b 10             	lea    0x10(%ebx),%ebp
  402c65:	33 db                	xor    %ebx,%ebx
  402c67:	33 c9                	xor    %ecx,%ecx
  402c69:	33 d2                	xor    %edx,%edx
  402c6b:	33 f6                	xor    %esi,%esi
  402c6d:	33 ff                	xor    %edi,%edi
  402c6f:	ff d0                	call   *%eax
  402c71:	5d                   	pop    %ebp
  402c72:	5e                   	pop    %esi
  402c73:	8b 5d 0c             	mov    0xc(%ebp),%ebx
  402c76:	0b c0                	or     %eax,%eax
  402c78:	74 4c                	je     0x402cc6
  402c7a:	78 58                	js     0x402cd4
  402c7c:	6a 01                	push   $0x1
  402c7e:	ff 75 08             	push   0x8(%ebp)
  402c81:	e8 9b 00 00 00       	call   0x402d21
  402c86:	83 c4 08             	add    $0x8,%esp
  402c89:	8b 7b 08             	mov    0x8(%ebx),%edi
  402c8c:	53                   	push   %ebx
  402c8d:	e8 ce 00 00 00       	call   0x402d60
  402c92:	83 c4 04             	add    $0x4,%esp
  402c95:	8d 6b 10             	lea    0x10(%ebx),%ebp
  402c98:	56                   	push   %esi
  402c99:	53                   	push   %ebx
  402c9a:	e8 26 01 00 00       	call   0x402dc5
  402c9f:	83 c4 08             	add    $0x8,%esp
  402ca2:	8d 0c 76             	lea    (%esi,%esi,2),%ecx
  402ca5:	6a 01                	push   $0x1
  402ca7:	8b 44 8f 08          	mov    0x8(%edi,%ecx,4),%eax
  402cab:	e8 c5 01 00 00       	call   0x402e75
  402cb0:	8b 04 8f             	mov    (%edi,%ecx,4),%eax
  402cb3:	89 43 0c             	mov    %eax,0xc(%ebx)
  402cb6:	8b 44 8f 08          	mov    0x8(%edi,%ecx,4),%eax
  402cba:	33 db                	xor    %ebx,%ebx
  402cbc:	33 c9                	xor    %ecx,%ecx
  402cbe:	33 d2                	xor    %edx,%edx
  402cc0:	33 f6                	xor    %esi,%esi
  402cc2:	33 ff                	xor    %edi,%edi
  402cc4:	ff d0                	call   *%eax
  402cc6:	8b 7b 08             	mov    0x8(%ebx),%edi
  402cc9:	8d 0c 76             	lea    (%esi,%esi,2),%ecx
  402ccc:	8b 34 8f             	mov    (%edi,%ecx,4),%esi
  402ccf:	e9 78 ff ff ff       	jmp    0x402c4c
  402cd4:	b8 00 00 00 00       	mov    $0x0,%eax
  402cd9:	eb 23                	jmp    0x402cfe
  402cdb:	8b 45 08             	mov    0x8(%ebp),%eax
  402cde:	83 48 04 08          	orl    $0x8,0x4(%eax)
  402ce2:	b8 01 00 00 00       	mov    $0x1,%eax
  402ce7:	eb 15                	jmp    0x402cfe
  402ce9:	55                   	push   %ebp
  402cea:	8d 6b 10             	lea    0x10(%ebx),%ebp
  402ced:	6a ff                	push   $0xffffffff
  402cef:	53                   	push   %ebx
  402cf0:	e8 d0 00 00 00       	call   0x402dc5
  402cf5:	83 c4 08             	add    $0x8,%esp
  402cf8:	5d                   	pop    %ebp
  402cf9:	b8 01 00 00 00       	mov    $0x1,%eax
  402cfe:	5d                   	pop    %ebp
  402cff:	5f                   	pop    %edi
  402d00:	5e                   	pop    %esi
  402d01:	5b                   	pop    %ebx
  402d02:	8b e5                	mov    %ebp,%esp
  402d04:	5d                   	pop    %ebp
  402d05:	c3                   	ret
  402d06:	55                   	push   %ebp
  402d07:	8b 4c 24 08          	mov    0x8(%esp),%ecx
  402d0b:	8b 29                	mov    (%ecx),%ebp
  402d0d:	8b 41 1c             	mov    0x1c(%ecx),%eax
  402d10:	50                   	push   %eax
  402d11:	8b 41 18             	mov    0x18(%ecx),%eax
  402d14:	50                   	push   %eax
  402d15:	e8 ab 00 00 00       	call   0x402dc5
  402d1a:	83 c4 08             	add    $0x8,%esp
  402d1d:	5d                   	pop    %ebp
  402d1e:	c2 04 00             	ret    $0x4
  402d21:	55                   	push   %ebp
  402d22:	8b ec                	mov    %esp,%ebp
  402d24:	56                   	push   %esi
  402d25:	8b 75 08             	mov    0x8(%ebp),%esi
  402d28:	81 3e 63 73 6d e0    	cmpl   $0xe06d7363,(%esi)
  402d2e:	75 24                	jne    0x402d54
  402d30:	83 3d ec af 41 00 00 	cmpl   $0x0,0x41afec
  402d37:	74 1b                	je     0x402d54
  402d39:	68 ec af 41 00       	push   $0x41afec
  402d3e:	e8 dd 05 00 00       	call   0x403320
  402d43:	59                   	pop    %ecx
  402d44:	85 c0                	test   %eax,%eax
  402d46:	74 0c                	je     0x402d54
  402d48:	ff 75 0c             	push   0xc(%ebp)
  402d4b:	56                   	push   %esi
  402d4c:	ff 15 ec af 41 00    	call   *0x41afec
  402d52:	59                   	pop    %ecx
  402d53:	59                   	pop    %ecx
  402d54:	5e                   	pop    %esi
  402d55:	5d                   	pop    %ebp
  402d56:	c3                   	ret
  402d57:	cc                   	int3
  402d58:	cc                   	int3
  402d59:	cc                   	int3
  402d5a:	cc                   	int3
  402d5b:	cc                   	int3
  402d5c:	cc                   	int3
  402d5d:	cc                   	int3
  402d5e:	cc                   	int3
  402d5f:	cc                   	int3
  402d60:	55                   	push   %ebp
  402d61:	8b ec                	mov    %esp,%ebp
  402d63:	53                   	push   %ebx
  402d64:	56                   	push   %esi
  402d65:	57                   	push   %edi
  402d66:	55                   	push   %ebp
  402d67:	6a 00                	push   $0x0
  402d69:	6a 00                	push   $0x0
  402d6b:	68 78 2d 40 00       	push   $0x402d78
  402d70:	ff 75 08             	push   0x8(%ebp)
  402d73:	e8 fe 0a 00 00       	call   0x403876
  402d78:	5d                   	pop    %ebp
  402d79:	5f                   	pop    %edi
  402d7a:	5e                   	pop    %esi
  402d7b:	5b                   	pop    %ebx
  402d7c:	8b e5                	mov    %ebp,%esp
  402d7e:	5d                   	pop    %ebp
  402d7f:	c3                   	ret
  402d80:	8b 4c 24 04          	mov    0x4(%esp),%ecx
  402d84:	f7 41 04 06 00 00 00 	testl  $0x6,0x4(%ecx)
  402d8b:	b8 01 00 00 00       	mov    $0x1,%eax
  402d90:	74 32                	je     0x402dc4
  402d92:	8b 44 24 14          	mov    0x14(%esp),%eax
  402d96:	8b 48 fc             	mov    -0x4(%eax),%ecx
  402d99:	33 c8                	xor    %eax,%ecx
  402d9b:	e8 71 06 00 00       	call   0x403411
  402da0:	55                   	push   %ebp
  402da1:	8b 68 10             	mov    0x10(%eax),%ebp
  402da4:	8b 50 28             	mov    0x28(%eax),%edx
  402da7:	52                   	push   %edx
  402da8:	8b 50 24             	mov    0x24(%eax),%edx
  402dab:	52                   	push   %edx
  402dac:	e8 14 00 00 00       	call   0x402dc5
  402db1:	83 c4 08             	add    $0x8,%esp
  402db4:	5d                   	pop    %ebp
  402db5:	8b 44 24 08          	mov    0x8(%esp),%eax
  402db9:	8b 54 24 10          	mov    0x10(%esp),%edx
  402dbd:	89 02                	mov    %eax,(%edx)
  402dbf:	b8 03 00 00 00       	mov    $0x3,%eax
  402dc4:	c3                   	ret
  402dc5:	53                   	push   %ebx
  402dc6:	56                   	push   %esi
  402dc7:	57                   	push   %edi
  402dc8:	8b 44 24 10          	mov    0x10(%esp),%eax
  402dcc:	55                   	push   %ebp
  402dcd:	50                   	push   %eax
  402dce:	6a fe                	push   $0xfffffffe
  402dd0:	68 80 2d 40 00       	push   $0x402d80
  402dd5:	64 ff 35 00 00 00 00 	push   %fs:0x0
  402ddc:	a1 d0 a7 41 00       	mov    0x41a7d0,%eax
  402de1:	33 c4                	xor    %esp,%eax
  402de3:	50                   	push   %eax
  402de4:	8d 44 24 04          	lea    0x4(%esp),%eax
  402de8:	64 a3 00 00 00 00    	mov    %eax,%fs:0x0
  402dee:	8b 44 24 28          	mov    0x28(%esp),%eax
  402df2:	8b 58 08             	mov    0x8(%eax),%ebx
  402df5:	8b 70 0c             	mov    0xc(%eax),%esi
  402df8:	83 fe ff             	cmp    $0xffffffff,%esi
  402dfb:	74 3a                	je     0x402e37
  402dfd:	83 7c 24 2c ff       	cmpl   $0xffffffff,0x2c(%esp)
  402e02:	74 06                	je     0x402e0a
  402e04:	3b 74 24 2c          	cmp    0x2c(%esp),%esi
  402e08:	76 2d                	jbe    0x402e37
  402e0a:	8d 34 76             	lea    (%esi,%esi,2),%esi
  402e0d:	8b 0c b3             	mov    (%ebx,%esi,4),%ecx
  402e10:	89 4c 24 0c          	mov    %ecx,0xc(%esp)
  402e14:	89 48 0c             	mov    %ecx,0xc(%eax)
  402e17:	83 7c b3 04 00       	cmpl   $0x0,0x4(%ebx,%esi,4)
  402e1c:	75 17                	jne    0x402e35
  402e1e:	68 01 01 00 00       	push   $0x101
  402e23:	8b 44 b3 08          	mov    0x8(%ebx,%esi,4),%eax
  402e27:	e8 49 00 00 00       	call   0x402e75
  402e2c:	8b 44 b3 08          	mov    0x8(%ebx,%esi,4),%eax
  402e30:	e8 5f 00 00 00       	call   0x402e94
  402e35:	eb b7                	jmp    0x402dee
  402e37:	8b 4c 24 04          	mov    0x4(%esp),%ecx
  402e3b:	64 89 0d 00 00 00 00 	mov    %ecx,%fs:0x0
  402e42:	83 c4 18             	add    $0x18,%esp
  402e45:	5f                   	pop    %edi
  402e46:	5e                   	pop    %esi
  402e47:	5b                   	pop    %ebx
  402e48:	c3                   	ret
  402e49:	33 c0                	xor    %eax,%eax
  402e4b:	64 8b 0d 00 00 00 00 	mov    %fs:0x0,%ecx
  402e52:	81 79 04 80 2d 40 00 	cmpl   $0x402d80,0x4(%ecx)
  402e59:	75 10                	jne    0x402e6b
  402e5b:	8b 51 0c             	mov    0xc(%ecx),%edx
  402e5e:	8b 52 0c             	mov    0xc(%edx),%edx
  402e61:	39 51 08             	cmp    %edx,0x8(%ecx)
  402e64:	75 05                	jne    0x402e6b
  402e66:	b8 01 00 00 00       	mov    $0x1,%eax
  402e6b:	c3                   	ret
  402e6c:	53                   	push   %ebx
  402e6d:	51                   	push   %ecx
  402e6e:	bb c0 a7 41 00       	mov    $0x41a7c0,%ebx
  402e73:	eb 0b                	jmp    0x402e80
  402e75:	53                   	push   %ebx
  402e76:	51                   	push   %ecx
  402e77:	bb c0 a7 41 00       	mov    $0x41a7c0,%ebx
  402e7c:	8b 4c 24 0c          	mov    0xc(%esp),%ecx
  402e80:	89 4b 08             	mov    %ecx,0x8(%ebx)
  402e83:	89 43 04             	mov    %eax,0x4(%ebx)
  402e86:	89 6b 0c             	mov    %ebp,0xc(%ebx)
  402e89:	55                   	push   %ebp
  402e8a:	51                   	push   %ecx
  402e8b:	50                   	push   %eax
  402e8c:	58                   	pop    %eax
  402e8d:	59                   	pop    %ecx
  402e8e:	5d                   	pop    %ebp
  402e8f:	59                   	pop    %ecx
  402e90:	5b                   	pop    %ebx
  402e91:	c2 04 00             	ret    $0x4
  402e94:	ff d0                	call   *%eax
  402e96:	c3                   	ret
  402e97:	cc                   	int3
  402e98:	cc                   	int3
  402e99:	cc                   	int3
  402e9a:	cc                   	int3
  402e9b:	cc                   	int3
  402e9c:	cc                   	int3
  402e9d:	cc                   	int3
  402e9e:	cc                   	int3
  402e9f:	cc                   	int3
  402ea0:	55                   	push   %ebp
  402ea1:	8b ec                	mov    %esp,%ebp
  402ea3:	8b 4d 10             	mov    0x10(%ebp),%ecx
  402ea6:	33 c0                	xor    %eax,%eax
  402ea8:	53                   	push   %ebx
  402ea9:	56                   	push   %esi
  402eaa:	83 ca ff             	or     $0xffffffff,%edx
  402ead:	57                   	push   %edi
  402eae:	83 f9 ff             	cmp    $0xffffffff,%ecx
  402eb1:	0f 84 96 00 00 00    	je     0x402f4d
  402eb7:	8b 7d 08             	mov    0x8(%ebp),%edi
  402eba:	8d 9b 00 00 00 00    	lea    0x0(%ebx),%ebx
  402ec0:	8b 5d 0c             	mov    0xc(%ebp),%ebx
  402ec3:	8d 0c 49             	lea    (%ecx,%ecx,2),%ecx
  402ec6:	8b 74 8b 08          	mov    0x8(%ebx,%ecx,4),%esi
  402eca:	8d 1c 8b             	lea    (%ebx,%ecx,4),%ebx
  402ecd:	2b f7                	sub    %edi,%esi
  402ecf:	81 e6 00 f0 ff ff    	and    $0xfffff000,%esi
  402ed5:	3b f2                	cmp    %edx,%esi
  402ed7:	74 2d                	je     0x402f06
  402ed9:	85 c0                	test   %eax,%eax
  402edb:	74 10                	je     0x402eed
  402edd:	8b 50 0c             	mov    0xc(%eax),%edx
  402ee0:	3b f2                	cmp    %edx,%esi
  402ee2:	72 09                	jb     0x402eed
  402ee4:	8b 48 08             	mov    0x8(%eax),%ecx
  402ee7:	03 ca                	add    %edx,%ecx
  402ee9:	3b f1                	cmp    %ecx,%esi
  402eeb:	72 17                	jb     0x402f04
  402eed:	56                   	push   %esi
  402eee:	57                   	push   %edi
  402eef:	e8 dc 03 00 00       	call   0x4032d0
  402ef4:	83 c4 08             	add    $0x8,%esp
  402ef7:	85 c0                	test   %eax,%eax
  402ef9:	74 5c                	je     0x402f57
  402efb:	f7 40 24 00 00 00 20 	testl  $0x20000000,0x24(%eax)
  402f02:	74 53                	je     0x402f57
  402f04:	8b d6                	mov    %esi,%edx
  402f06:	8b 73 04             	mov    0x4(%ebx),%esi
  402f09:	85 f6                	test   %esi,%esi
  402f0b:	74 35                	je     0x402f42
  402f0d:	2b f7                	sub    %edi,%esi
  402f0f:	81 e6 00 f0 ff ff    	and    $0xfffff000,%esi
  402f15:	3b f2                	cmp    %edx,%esi
  402f17:	74 29                	je     0x402f42
  402f19:	8b 50 0c             	mov    0xc(%eax),%edx
  402f1c:	3b f2                	cmp    %edx,%esi
  402f1e:	72 09                	jb     0x402f29
  402f20:	8b 48 08             	mov    0x8(%eax),%ecx
  402f23:	03 ca                	add    %edx,%ecx
  402f25:	3b f1                	cmp    %ecx,%esi
  402f27:	72 17                	jb     0x402f40
  402f29:	56                   	push   %esi
  402f2a:	57                   	push   %edi
  402f2b:	e8 a0 03 00 00       	call   0x4032d0
  402f30:	83 c4 08             	add    $0x8,%esp
  402f33:	85 c0                	test   %eax,%eax
  402f35:	74 20                	je     0x402f57
  402f37:	f7 40 24 00 00 00 20 	testl  $0x20000000,0x24(%eax)
  402f3e:	74 17                	je     0x402f57
  402f40:	8b d6                	mov    %esi,%edx
  402f42:	8b 0b                	mov    (%ebx),%ecx
  402f44:	83 f9 ff             	cmp    $0xffffffff,%ecx
  402f47:	0f 85 73 ff ff ff    	jne    0x402ec0
  402f4d:	5f                   	pop    %edi
  402f4e:	5e                   	pop    %esi
  402f4f:	b8 01 00 00 00       	mov    $0x1,%eax
  402f54:	5b                   	pop    %ebx
  402f55:	5d                   	pop    %ebp
  402f56:	c3                   	ret
  402f57:	5f                   	pop    %edi
  402f58:	5e                   	pop    %esi
  402f59:	33 c0                	xor    %eax,%eax
  402f5b:	5b                   	pop    %ebx
  402f5c:	5d                   	pop    %ebp
  402f5d:	c3                   	ret
  402f5e:	cc                   	int3
  402f5f:	cc                   	int3
  402f60:	55                   	push   %ebp
  402f61:	8b ec                	mov    %esp,%ebp
  402f63:	6a fe                	push   $0xfffffffe
  402f65:	68 d0 72 40 00       	push   $0x4072d0
  402f6a:	68 20 34 40 00       	push   $0x403420
  402f6f:	64 a1 00 00 00 00    	mov    %fs:0x0,%eax
  402f75:	50                   	push   %eax
  402f76:	83 ec 38             	sub    $0x38,%esp
  402f79:	53                   	push   %ebx
  402f7a:	56                   	push   %esi
  402f7b:	57                   	push   %edi
  402f7c:	a1 d0 a7 41 00       	mov    0x41a7d0,%eax
  402f81:	31 45 f8             	xor    %eax,-0x8(%ebp)
  402f84:	33 c5                	xor    %ebp,%eax
  402f86:	50                   	push   %eax
  402f87:	8d 45 f0             	lea    -0x10(%ebp),%eax
  402f8a:	64 a3 00 00 00 00    	mov    %eax,%fs:0x0
  402f90:	89 65 e8             	mov    %esp,-0x18(%ebp)
  402f93:	8b 7d 08             	mov    0x8(%ebp),%edi
  402f96:	8b 5f 08             	mov    0x8(%edi),%ebx
  402f99:	89 5d dc             	mov    %ebx,-0x24(%ebp)
  402f9c:	89 5d d4             	mov    %ebx,-0x2c(%ebp)
  402f9f:	f6 c3 03             	test   $0x3,%bl
  402fa2:	74 14                	je     0x402fb8
  402fa4:	33 c0                	xor    %eax,%eax
  402fa6:	8b 4d f0             	mov    -0x10(%ebp),%ecx
  402fa9:	64 89 0d 00 00 00 00 	mov    %ecx,%fs:0x0
  402fb0:	59                   	pop    %ecx
  402fb1:	5f                   	pop    %edi
  402fb2:	5e                   	pop    %esi
  402fb3:	5b                   	pop    %ebx
  402fb4:	8b e5                	mov    %ebp,%esp
  402fb6:	5d                   	pop    %ebp
  402fb7:	c3                   	ret
  402fb8:	64 a1 18 00 00 00    	mov    %fs:0x18,%eax
  402fbe:	8b 48 08             	mov    0x8(%eax),%ecx
  402fc1:	89 4d d8             	mov    %ecx,-0x28(%ebp)
  402fc4:	3b d9                	cmp    %ecx,%ebx
  402fc6:	72 05                	jb     0x402fcd
  402fc8:	3b 58 04             	cmp    0x4(%eax),%ebx
  402fcb:	72 d7                	jb     0x402fa4
  402fcd:	8b 57 0c             	mov    0xc(%edi),%edx
  402fd0:	89 55 e4             	mov    %edx,-0x1c(%ebp)
  402fd3:	83 fa ff             	cmp    $0xffffffff,%edx
  402fd6:	0f 84 cf 02 00 00    	je     0x4032ab
  402fdc:	c7 45 e0 00 00 00 00 	movl   $0x0,-0x20(%ebp)
  402fe3:	33 c9                	xor    %ecx,%ecx
  402fe5:	8b c3                	mov    %ebx,%eax
  402fe7:	8b 30                	mov    (%eax),%esi
  402fe9:	83 fe ff             	cmp    $0xffffffff,%esi
  402fec:	74 04                	je     0x402ff2
  402fee:	3b f1                	cmp    %ecx,%esi
  402ff0:	73 b2                	jae    0x402fa4
  402ff2:	83 78 04 00          	cmpl   $0x0,0x4(%eax)
  402ff6:	74 0a                	je     0x403002
  402ff8:	be 01 00 00 00       	mov    $0x1,%esi
  402ffd:	89 75 e0             	mov    %esi,-0x20(%ebp)
  403000:	eb 03                	jmp    0x403005
  403002:	8b 75 e0             	mov    -0x20(%ebp),%esi
  403005:	41                   	inc    %ecx
  403006:	83 c0 0c             	add    $0xc,%eax
  403009:	3b ca                	cmp    %edx,%ecx
  40300b:	76 da                	jbe    0x402fe7
  40300d:	85 f6                	test   %esi,%esi
  40300f:	74 0c                	je     0x40301d
  403011:	8b 47 f8             	mov    -0x8(%edi),%eax
  403014:	3b 45 d8             	cmp    -0x28(%ebp),%eax
  403017:	72 8b                	jb     0x402fa4
  403019:	3b c7                	cmp    %edi,%eax
  40301b:	73 87                	jae    0x402fa4
  40301d:	8b fb                	mov    %ebx,%edi
  40301f:	81 e7 00 f0 ff ff    	and    $0xfffff000,%edi
  403025:	89 7d d8             	mov    %edi,-0x28(%ebp)
  403028:	33 f6                	xor    %esi,%esi
  40302a:	8b 0d c0 ac 41 00    	mov    0x41acc0,%ecx
  403030:	3b f1                	cmp    %ecx,%esi
  403032:	0f 8d 40 01 00 00    	jge    0x403178
  403038:	8b 04 f5 40 ac 41 00 	mov    0x41ac40(,%esi,8),%eax
  40303f:	89 45 e0             	mov    %eax,-0x20(%ebp)
  403042:	8b 1c f5 44 ac 41 00 	mov    0x41ac44(,%esi,8),%ebx
  403049:	3b c7                	cmp    %edi,%eax
  40304b:	0f 85 1e 01 00 00    	jne    0x40316f
  403051:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%ebp)
  403058:	53                   	push   %ebx
  403059:	e8 82 03 00 00       	call   0x4033e0
  40305e:	83 c4 04             	add    $0x4,%esp
  403061:	85 c0                	test   %eax,%eax
  403063:	0f 84 cf 00 00 00    	je     0x403138
  403069:	ff 75 e4             	push   -0x1c(%ebp)
  40306c:	ff 75 dc             	push   -0x24(%ebp)
  40306f:	53                   	push   %ebx
  403070:	e8 2b fe ff ff       	call   0x402ea0
  403075:	83 c4 0c             	add    $0xc,%esp
  403078:	85 c0                	test   %eax,%eax
  40307a:	0f 84 b8 00 00 00    	je     0x403138
  403080:	8b 45 08             	mov    0x8(%ebp),%eax
  403083:	8b 40 04             	mov    0x4(%eax),%eax
  403086:	2b c3                	sub    %ebx,%eax
  403088:	50                   	push   %eax
  403089:	53                   	push   %ebx
  40308a:	e8 41 02 00 00       	call   0x4032d0
  40308f:	83 c4 08             	add    $0x8,%esp
  403092:	85 c0                	test   %eax,%eax
  403094:	0f 84 9e 00 00 00    	je     0x403138
  40309a:	c7 45 fc fe ff ff ff 	movl   $0xfffffffe,-0x4(%ebp)
  4030a1:	85 f6                	test   %esi,%esi
  4030a3:	0f 8e 02 02 00 00    	jle    0x4032ab
  4030a9:	b8 01 00 00 00       	mov    $0x1,%eax
  4030ae:	b9 c4 ac 41 00       	mov    $0x41acc4,%ecx
  4030b3:	87 01                	xchg   %eax,(%ecx)
  4030b5:	85 c0                	test   %eax,%eax
  4030b7:	0f 85 ee 01 00 00    	jne    0x4032ab
  4030bd:	39 3c f5 40 ac 41 00 	cmp    %edi,0x41ac40(,%esi,8)
  4030c4:	74 3f                	je     0x403105
  4030c6:	a1 c0 ac 41 00       	mov    0x41acc0,%eax
  4030cb:	8d 70 ff             	lea    -0x1(%eax),%esi
  4030ce:	85 f6                	test   %esi,%esi
  4030d0:	78 0c                	js     0x4030de
  4030d2:	39 3c f5 40 ac 41 00 	cmp    %edi,0x41ac40(,%esi,8)
  4030d9:	74 1a                	je     0x4030f5
  4030db:	4e                   	dec    %esi
  4030dc:	79 f4                	jns    0x4030d2
  4030de:	8b 7d e0             	mov    -0x20(%ebp),%edi
  4030e1:	85 f6                	test   %esi,%esi
  4030e3:	79 25                	jns    0x40310a
  4030e5:	83 f8 10             	cmp    $0x10,%eax
  4030e8:	7d 06                	jge    0x4030f0
  4030ea:	40                   	inc    %eax
  4030eb:	a3 c0 ac 41 00       	mov    %eax,0x41acc0
  4030f0:	8d 70 ff             	lea    -0x1(%eax),%esi
  4030f3:	eb 13                	jmp    0x403108
  4030f5:	8b 3c f5 40 ac 41 00 	mov    0x41ac40(,%esi,8),%edi
  4030fc:	8b 1c f5 44 ac 41 00 	mov    0x41ac44(,%esi,8),%ebx
  403103:	eb dc                	jmp    0x4030e1
  403105:	8b 7d e0             	mov    -0x20(%ebp),%edi
  403108:	85 f6                	test   %esi,%esi
  40310a:	7e 23                	jle    0x40312f
  40310c:	85 f6                	test   %esi,%esi
  40310e:	78 1f                	js     0x40312f
  403110:	ba 40 ac 41 00       	mov    $0x41ac40,%edx
  403115:	46                   	inc    %esi
  403116:	8b 02                	mov    (%edx),%eax
  403118:	8b 4a 04             	mov    0x4(%edx),%ecx
  40311b:	89 3a                	mov    %edi,(%edx)
  40311d:	89 5a 04             	mov    %ebx,0x4(%edx)
  403120:	8b f8                	mov    %eax,%edi
  403122:	8b d9                	mov    %ecx,%ebx
  403124:	8d 52 08             	lea    0x8(%edx),%edx
  403127:	4e                   	dec    %esi
  403128:	75 ec                	jne    0x403116
  40312a:	b9 c4 ac 41 00       	mov    $0x41acc4,%ecx
  40312f:	33 c0                	xor    %eax,%eax
  403131:	87 01                	xchg   %eax,(%ecx)
  403133:	e9 73 01 00 00       	jmp    0x4032ab
  403138:	c7 45 fc fe ff ff ff 	movl   $0xfffffffe,-0x4(%ebp)
  40313f:	8b 5d dc             	mov    -0x24(%ebp),%ebx
  403142:	8b 75 e4             	mov    -0x1c(%ebp),%esi
  403145:	eb 33                	jmp    0x40317a
  403147:	8b 45 ec             	mov    -0x14(%ebp),%eax
  40314a:	8b 00                	mov    (%eax),%eax
  40314c:	33 c9                	xor    %ecx,%ecx
  40314e:	81 38 05 00 00 c0    	cmpl   $0xc0000005,(%eax)
  403154:	0f 94 c1             	sete   %cl
  403157:	8b c1                	mov    %ecx,%eax
  403159:	c3                   	ret
  40315a:	8b 65 e8             	mov    -0x18(%ebp),%esp
  40315d:	c7 45 fc fe ff ff ff 	movl   $0xfffffffe,-0x4(%ebp)
  403164:	8b 5d d4             	mov    -0x2c(%ebp),%ebx
  403167:	8b 75 e4             	mov    -0x1c(%ebp),%esi
  40316a:	8b 7d d8             	mov    -0x28(%ebp),%edi
  40316d:	eb 0b                	jmp    0x40317a
  40316f:	46                   	inc    %esi
  403170:	8b 5d dc             	mov    -0x24(%ebp),%ebx
  403173:	e9 b8 fe ff ff       	jmp    0x403030
  403178:	8b f2                	mov    %edx,%esi
  40317a:	6a 1c                	push   $0x1c
  40317c:	8d 45 b8             	lea    -0x48(%ebp),%eax
  40317f:	50                   	push   %eax
  403180:	53                   	push   %ebx
  403181:	ff 15 4c 60 40 00    	call   *0x40604c
  403187:	85 c0                	test   %eax,%eax
  403189:	0f 84 1c 01 00 00    	je     0x4032ab
  40318f:	81 7d d0 00 00 00 01 	cmpl   $0x1000000,-0x30(%ebp)
  403196:	74 15                	je     0x4031ad
  403198:	83 c8 ff             	or     $0xffffffff,%eax
  40319b:	8b 4d f0             	mov    -0x10(%ebp),%ecx
  40319e:	64 89 0d 00 00 00 00 	mov    %ecx,%fs:0x0
  4031a5:	59                   	pop    %ecx
  4031a6:	5f                   	pop    %edi
  4031a7:	5e                   	pop    %esi
  4031a8:	5b                   	pop    %ebx
  4031a9:	8b e5                	mov    %ebp,%esp
  4031ab:	5d                   	pop    %ebp
  4031ac:	c3                   	ret
  4031ad:	8b 45 bc             	mov    -0x44(%ebp),%eax
  4031b0:	89 45 dc             	mov    %eax,-0x24(%ebp)
  4031b3:	50                   	push   %eax
  4031b4:	e8 27 02 00 00       	call   0x4033e0
  4031b9:	83 c4 04             	add    $0x4,%esp
  4031bc:	85 c0                	test   %eax,%eax
  4031be:	74 d8                	je     0x403198
  4031c0:	f6 45 cc cc          	testb  $0xcc,-0x34(%ebp)
  4031c4:	74 26                	je     0x4031ec
  4031c6:	8b c3                	mov    %ebx,%eax
  4031c8:	8b 4d dc             	mov    -0x24(%ebp),%ecx
  4031cb:	2b c1                	sub    %ecx,%eax
  4031cd:	50                   	push   %eax
  4031ce:	51                   	push   %ecx
  4031cf:	e8 fc 00 00 00       	call   0x4032d0
  4031d4:	83 c4 08             	add    $0x8,%esp
  4031d7:	85 c0                	test   %eax,%eax
  4031d9:	0f 84 c5 fd ff ff    	je     0x402fa4
  4031df:	f7 40 24 00 00 00 80 	testl  $0x80000000,0x24(%eax)
  4031e6:	0f 85 b8 fd ff ff    	jne    0x402fa4
  4031ec:	56                   	push   %esi
  4031ed:	53                   	push   %ebx
  4031ee:	8b 5d dc             	mov    -0x24(%ebp),%ebx
  4031f1:	53                   	push   %ebx
  4031f2:	e8 a9 fc ff ff       	call   0x402ea0
  4031f7:	83 c4 0c             	add    $0xc,%esp
  4031fa:	85 c0                	test   %eax,%eax
  4031fc:	0f 84 a2 fd ff ff    	je     0x402fa4
  403202:	8b 45 08             	mov    0x8(%ebp),%eax
  403205:	8b 40 04             	mov    0x4(%eax),%eax
  403208:	2b c3                	sub    %ebx,%eax
  40320a:	50                   	push   %eax
  40320b:	53                   	push   %ebx
  40320c:	e8 bf 00 00 00       	call   0x4032d0
  403211:	83 c4 08             	add    $0x8,%esp
  403214:	85 c0                	test   %eax,%eax
  403216:	0f 84 88 fd ff ff    	je     0x402fa4
  40321c:	b8 01 00 00 00       	mov    $0x1,%eax
  403221:	be c4 ac 41 00       	mov    $0x41acc4,%esi
  403226:	87 06                	xchg   %eax,(%esi)
  403228:	85 c0                	test   %eax,%eax
  40322a:	0f 85 7b 00 00 00    	jne    0x4032ab
  403230:	8b 15 c0 ac 41 00    	mov    0x41acc0,%edx
  403236:	8b ca                	mov    %edx,%ecx
  403238:	85 d2                	test   %edx,%edx
  40323a:	7e 13                	jle    0x40324f
  40323c:	8d 04 d5 38 ac 41 00 	lea    0x41ac38(,%edx,8),%eax
  403243:	39 38                	cmp    %edi,(%eax)
  403245:	74 08                	je     0x40324f
  403247:	49                   	dec    %ecx
  403248:	83 e8 08             	sub    $0x8,%eax
  40324b:	85 c9                	test   %ecx,%ecx
  40324d:	7f f4                	jg     0x403243
  40324f:	85 c9                	test   %ecx,%ecx
  403251:	75 4a                	jne    0x40329d
  403253:	83 fa 0f             	cmp    $0xf,%edx
  403256:	8d 41 0f             	lea    0xf(%ecx),%eax
  403259:	7f 02                	jg     0x40325d
  40325b:	8b c2                	mov    %edx,%eax
  40325d:	85 c0                	test   %eax,%eax
  40325f:	78 2e                	js     0x40328f
  403261:	bb 40 ac 41 00       	mov    $0x41ac40,%ebx
  403266:	8d 50 01             	lea    0x1(%eax),%edx
  403269:	8b 75 bc             	mov    -0x44(%ebp),%esi
  40326c:	8d 64 24 00          	lea    0x0(%esp),%esp
  403270:	8b 03                	mov    (%ebx),%eax
  403272:	8b 4b 04             	mov    0x4(%ebx),%ecx
  403275:	89 3b                	mov    %edi,(%ebx)
  403277:	89 73 04             	mov    %esi,0x4(%ebx)
  40327a:	8b f8                	mov    %eax,%edi
  40327c:	8b f1                	mov    %ecx,%esi
  40327e:	8d 5b 08             	lea    0x8(%ebx),%ebx
  403281:	4a                   	dec    %edx
  403282:	75 ec                	jne    0x403270
  403284:	8b 15 c0 ac 41 00    	mov    0x41acc0,%edx
  40328a:	be c4 ac 41 00       	mov    $0x41acc4,%esi
  40328f:	83 fa 10             	cmp    $0x10,%edx
  403292:	7d 13                	jge    0x4032a7
  403294:	42                   	inc    %edx
  403295:	89 15 c0 ac 41 00    	mov    %edx,0x41acc0
  40329b:	eb 0a                	jmp    0x4032a7
  40329d:	8b 45 bc             	mov    -0x44(%ebp),%eax
  4032a0:	89 04 cd 3c ac 41 00 	mov    %eax,0x41ac3c(,%ecx,8)
  4032a7:	33 c0                	xor    %eax,%eax
  4032a9:	87 06                	xchg   %eax,(%esi)
  4032ab:	b8 01 00 00 00       	mov    $0x1,%eax
  4032b0:	8b 4d f0             	mov    -0x10(%ebp),%ecx
  4032b3:	64 89 0d 00 00 00 00 	mov    %ecx,%fs:0x0
  4032ba:	59                   	pop    %ecx
  4032bb:	5f                   	pop    %edi
  4032bc:	5e                   	pop    %esi
  4032bd:	5b                   	pop    %ebx
  4032be:	8b e5                	mov    %ebp,%esp
  4032c0:	5d                   	pop    %ebp
  4032c1:	c3                   	ret
  4032c2:	cc                   	int3
  4032c3:	cc                   	int3
  4032c4:	cc                   	int3
  4032c5:	cc                   	int3
  4032c6:	cc                   	int3
  4032c7:	cc                   	int3
  4032c8:	cc                   	int3
  4032c9:	cc                   	int3
  4032ca:	cc                   	int3
  4032cb:	cc                   	int3
  4032cc:	cc                   	int3
  4032cd:	cc                   	int3
  4032ce:	cc                   	int3
  4032cf:	cc                   	int3
  4032d0:	55                   	push   %ebp
  4032d1:	8b ec                	mov    %esp,%ebp
  4032d3:	8b 45 08             	mov    0x8(%ebp),%eax
  4032d6:	33 d2                	xor    %edx,%edx
  4032d8:	53                   	push   %ebx
  4032d9:	56                   	push   %esi
  4032da:	57                   	push   %edi
  4032db:	8b 48 3c             	mov    0x3c(%eax),%ecx
  4032de:	03 c8                	add    %eax,%ecx
  4032e0:	0f b7 41 14          	movzwl 0x14(%ecx),%eax
  4032e4:	0f b7 59 06          	movzwl 0x6(%ecx),%ebx
  4032e8:	83 c0 18             	add    $0x18,%eax
  4032eb:	03 c1                	add    %ecx,%eax
  4032ed:	85 db                	test   %ebx,%ebx
  4032ef:	74 1b                	je     0x40330c
  4032f1:	8b 7d 0c             	mov    0xc(%ebp),%edi
  4032f4:	8b 70 0c             	mov    0xc(%eax),%esi
  4032f7:	3b fe                	cmp    %esi,%edi
  4032f9:	72 09                	jb     0x403304
  4032fb:	8b 48 08             	mov    0x8(%eax),%ecx
  4032fe:	03 ce                	add    %esi,%ecx
  403300:	3b f9                	cmp    %ecx,%edi
  403302:	72 0a                	jb     0x40330e
  403304:	42                   	inc    %edx
  403305:	83 c0 28             	add    $0x28,%eax
  403308:	3b d3                	cmp    %ebx,%edx
  40330a:	72 e8                	jb     0x4032f4
  40330c:	33 c0                	xor    %eax,%eax
  40330e:	5f                   	pop    %edi
  40330f:	5e                   	pop    %esi
  403310:	5b                   	pop    %ebx
  403311:	5d                   	pop    %ebp
  403312:	c3                   	ret
  403313:	cc                   	int3
  403314:	cc                   	int3
  403315:	cc                   	int3
  403316:	cc                   	int3
  403317:	cc                   	int3
  403318:	cc                   	int3
  403319:	cc                   	int3
  40331a:	cc                   	int3
  40331b:	cc                   	int3
  40331c:	cc                   	int3
  40331d:	cc                   	int3
  40331e:	cc                   	int3
  40331f:	cc                   	int3
  403320:	55                   	push   %ebp
  403321:	8b ec                	mov    %esp,%ebp
  403323:	6a fe                	push   $0xfffffffe
  403325:	68 f0 72 40 00       	push   $0x4072f0
  40332a:	68 20 34 40 00       	push   $0x403420
  40332f:	64 a1 00 00 00 00    	mov    %fs:0x0,%eax
  403335:	50                   	push   %eax
  403336:	83 ec 08             	sub    $0x8,%esp
  403339:	53                   	push   %ebx
  40333a:	56                   	push   %esi
  40333b:	57                   	push   %edi
  40333c:	a1 d0 a7 41 00       	mov    0x41a7d0,%eax
  403341:	31 45 f8             	xor    %eax,-0x8(%ebp)
  403344:	33 c5                	xor    %ebp,%eax
  403346:	50                   	push   %eax
  403347:	8d 45 f0             	lea    -0x10(%ebp),%eax
  40334a:	64 a3 00 00 00 00    	mov    %eax,%fs:0x0
  403350:	89 65 e8             	mov    %esp,-0x18(%ebp)
  403353:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%ebp)
  40335a:	68 00 00 40 00       	push   $0x400000
  40335f:	e8 7c 00 00 00       	call   0x4033e0
  403364:	83 c4 04             	add    $0x4,%esp
  403367:	85 c0                	test   %eax,%eax
  403369:	74 54                	je     0x4033bf
  40336b:	8b 45 08             	mov    0x8(%ebp),%eax
  40336e:	2d 00 00 40 00       	sub    $0x400000,%eax
  403373:	50                   	push   %eax
  403374:	68 00 00 40 00       	push   $0x400000
  403379:	e8 52 ff ff ff       	call   0x4032d0
  40337e:	83 c4 08             	add    $0x8,%esp
  403381:	85 c0                	test   %eax,%eax
  403383:	74 3a                	je     0x4033bf
  403385:	8b 40 24             	mov    0x24(%eax),%eax
  403388:	c1 e8 1f             	shr    $0x1f,%eax
  40338b:	f7 d0                	not    %eax
  40338d:	83 e0 01             	and    $0x1,%eax
  403390:	c7 45 fc fe ff ff ff 	movl   $0xfffffffe,-0x4(%ebp)
  403397:	8b 4d f0             	mov    -0x10(%ebp),%ecx
  40339a:	64 89 0d 00 00 00 00 	mov    %ecx,%fs:0x0
  4033a1:	59                   	pop    %ecx
  4033a2:	5f                   	pop    %edi
  4033a3:	5e                   	pop    %esi
  4033a4:	5b                   	pop    %ebx
  4033a5:	8b e5                	mov    %ebp,%esp
  4033a7:	5d                   	pop    %ebp
  4033a8:	c3                   	ret
  4033a9:	8b 45 ec             	mov    -0x14(%ebp),%eax
  4033ac:	8b 00                	mov    (%eax),%eax
  4033ae:	33 c9                	xor    %ecx,%ecx
  4033b0:	81 38 05 00 00 c0    	cmpl   $0xc0000005,(%eax)
  4033b6:	0f 94 c1             	sete   %cl
  4033b9:	8b c1                	mov    %ecx,%eax
  4033bb:	c3                   	ret
  4033bc:	8b 65 e8             	mov    -0x18(%ebp),%esp
  4033bf:	c7 45 fc fe ff ff ff 	movl   $0xfffffffe,-0x4(%ebp)
  4033c6:	33 c0                	xor    %eax,%eax
  4033c8:	8b 4d f0             	mov    -0x10(%ebp),%ecx
  4033cb:	64 89 0d 00 00 00 00 	mov    %ecx,%fs:0x0
  4033d2:	59                   	pop    %ecx
  4033d3:	5f                   	pop    %edi
  4033d4:	5e                   	pop    %esi
  4033d5:	5b                   	pop    %ebx
  4033d6:	8b e5                	mov    %ebp,%esp
  4033d8:	5d                   	pop    %ebp
  4033d9:	c3                   	ret
  4033da:	cc                   	int3
  4033db:	cc                   	int3
  4033dc:	cc                   	int3
  4033dd:	cc                   	int3
  4033de:	cc                   	int3
  4033df:	cc                   	int3
  4033e0:	55                   	push   %ebp
  4033e1:	8b ec                	mov    %esp,%ebp
  4033e3:	8b 45 08             	mov    0x8(%ebp),%eax
  4033e6:	b9 4d 5a 00 00       	mov    $0x5a4d,%ecx
  4033eb:	66 39 08             	cmp    %cx,(%eax)
  4033ee:	74 04                	je     0x4033f4
  4033f0:	33 c0                	xor    %eax,%eax
  4033f2:	5d                   	pop    %ebp
  4033f3:	c3                   	ret
  4033f4:	8b 48 3c             	mov    0x3c(%eax),%ecx
  4033f7:	03 c8                	add    %eax,%ecx
  4033f9:	33 c0                	xor    %eax,%eax
  4033fb:	81 39 50 45 00 00    	cmpl   $0x4550,(%ecx)
  403401:	75 0c                	jne    0x40340f
  403403:	ba 0b 01 00 00       	mov    $0x10b,%edx
  403408:	66 39 51 18          	cmp    %dx,0x18(%ecx)
  40340c:	0f 94 c0             	sete   %al
  40340f:	5d                   	pop    %ebp
  403410:	c3                   	ret
  403411:	3b 0d d0 a7 41 00    	cmp    0x41a7d0,%ecx
  403417:	75 02                	jne    0x40341b
  403419:	f3 c3                	repz ret
  40341b:	e9 d1 01 00 00       	jmp    0x4035f1
  403420:	55                   	push   %ebp
  403421:	8b ec                	mov    %esp,%ebp
  403423:	83 ec 18             	sub    $0x18,%esp
  403426:	53                   	push   %ebx
  403427:	8b 5d 0c             	mov    0xc(%ebp),%ebx
  40342a:	56                   	push   %esi
  40342b:	57                   	push   %edi
  40342c:	c6 45 ff 00          	movb   $0x0,-0x1(%ebp)
  403430:	8b 7b 08             	mov    0x8(%ebx),%edi
  403433:	8d 73 10             	lea    0x10(%ebx),%esi
  403436:	33 3d d0 a7 41 00    	xor    0x41a7d0,%edi
  40343c:	c7 45 f4 01 00 00 00 	movl   $0x1,-0xc(%ebp)
  403443:	8b 07                	mov    (%edi),%eax
  403445:	83 f8 fe             	cmp    $0xfffffffe,%eax
  403448:	74 0d                	je     0x403457
  40344a:	8b 4f 04             	mov    0x4(%edi),%ecx
  40344d:	03 ce                	add    %esi,%ecx
  40344f:	33 0c 30             	xor    (%eax,%esi,1),%ecx
  403452:	e8 ba ff ff ff       	call   0x403411
  403457:	8b 47 08             	mov    0x8(%edi),%eax
  40345a:	8b 4f 0c             	mov    0xc(%edi),%ecx
  40345d:	03 ce                	add    %esi,%ecx
  40345f:	33 0c 30             	xor    (%eax,%esi,1),%ecx
  403462:	e8 aa ff ff ff       	call   0x403411
  403467:	8b 45 08             	mov    0x8(%ebp),%eax
  40346a:	f6 40 04 66          	testb  $0x66,0x4(%eax)
  40346e:	0f 85 cf 00 00 00    	jne    0x403543
  403474:	89 45 e8             	mov    %eax,-0x18(%ebp)
  403477:	8b 45 10             	mov    0x10(%ebp),%eax
  40347a:	89 45 ec             	mov    %eax,-0x14(%ebp)
  40347d:	8d 45 e8             	lea    -0x18(%ebp),%eax
  403480:	89 43 fc             	mov    %eax,-0x4(%ebx)
  403483:	8b 43 0c             	mov    0xc(%ebx),%eax
  403486:	89 45 f8             	mov    %eax,-0x8(%ebp)
  403489:	83 f8 fe             	cmp    $0xfffffffe,%eax
  40348c:	0f 84 ed 00 00 00    	je     0x40357f
  403492:	8d 04 40             	lea    (%eax,%eax,2),%eax
  403495:	8d 40 04             	lea    0x4(%eax),%eax
  403498:	8b 4c 87 04          	mov    0x4(%edi,%eax,4),%ecx
  40349c:	8d 04 87             	lea    (%edi,%eax,4),%eax
  40349f:	8b 18                	mov    (%eax),%ebx
  4034a1:	89 45 f0             	mov    %eax,-0x10(%ebp)
  4034a4:	85 c9                	test   %ecx,%ecx
  4034a6:	74 7b                	je     0x403523
  4034a8:	8b d6                	mov    %esi,%edx
  4034aa:	e8 33 03 00 00       	call   0x4037e2
  4034af:	b1 01                	mov    $0x1,%cl
  4034b1:	88 4d ff             	mov    %cl,-0x1(%ebp)
  4034b4:	85 c0                	test   %eax,%eax
  4034b6:	0f 88 7e 00 00 00    	js     0x40353a
  4034bc:	7e 68                	jle    0x403526
  4034be:	8b 45 08             	mov    0x8(%ebp),%eax
  4034c1:	81 38 63 73 6d e0    	cmpl   $0xe06d7363,(%eax)
  4034c7:	75 28                	jne    0x4034f1
  4034c9:	83 3d ec af 41 00 00 	cmpl   $0x0,0x41afec
  4034d0:	74 1f                	je     0x4034f1
  4034d2:	68 ec af 41 00       	push   $0x41afec
  4034d7:	e8 44 fe ff ff       	call   0x403320
  4034dc:	83 c4 04             	add    $0x4,%esp
  4034df:	85 c0                	test   %eax,%eax
  4034e1:	74 0e                	je     0x4034f1
  4034e3:	6a 01                	push   $0x1
  4034e5:	ff 75 08             	push   0x8(%ebp)
  4034e8:	ff 15 ec af 41 00    	call   *0x41afec
  4034ee:	83 c4 08             	add    $0x8,%esp
  4034f1:	8b 55 08             	mov    0x8(%ebp),%edx
  4034f4:	8b 4d 0c             	mov    0xc(%ebp),%ecx
  4034f7:	e8 16 03 00 00       	call   0x403812
  4034fc:	8b 45 0c             	mov    0xc(%ebp),%eax
  4034ff:	8b 55 f8             	mov    -0x8(%ebp),%edx
  403502:	39 50 0c             	cmp    %edx,0xc(%eax)
  403505:	74 10                	je     0x403517
  403507:	68 d0 a7 41 00       	push   $0x41a7d0
  40350c:	56                   	push   %esi
  40350d:	8b c8                	mov    %eax,%ecx
  40350f:	e8 17 03 00 00       	call   0x40382b
  403514:	8b 45 0c             	mov    0xc(%ebp),%eax
  403517:	89 58 0c             	mov    %ebx,0xc(%eax)
  40351a:	8b 07                	mov    (%edi),%eax
  40351c:	83 f8 fe             	cmp    $0xfffffffe,%eax
  40351f:	74 75                	je     0x403596
  403521:	eb 66                	jmp    0x403589
  403523:	8a 4d ff             	mov    -0x1(%ebp),%cl
  403526:	89 5d f8             	mov    %ebx,-0x8(%ebp)
  403529:	8b c3                	mov    %ebx,%eax
  40352b:	83 fb fe             	cmp    $0xfffffffe,%ebx
  40352e:	0f 85 5e ff ff ff    	jne    0x403492
  403534:	84 c9                	test   %cl,%cl
  403536:	74 47                	je     0x40357f
  403538:	eb 21                	jmp    0x40355b
  40353a:	c7 45 f4 00 00 00 00 	movl   $0x0,-0xc(%ebp)
  403541:	eb 18                	jmp    0x40355b
  403543:	83 7b 0c fe          	cmpl   $0xfffffffe,0xc(%ebx)
  403547:	74 36                	je     0x40357f
  403549:	68 d0 a7 41 00       	push   $0x41a7d0
  40354e:	56                   	push   %esi
  40354f:	8b cb                	mov    %ebx,%ecx
  403551:	ba fe ff ff ff       	mov    $0xfffffffe,%edx
  403556:	e8 d0 02 00 00       	call   0x40382b
  40355b:	8b 07                	mov    (%edi),%eax
  40355d:	83 f8 fe             	cmp    $0xfffffffe,%eax
  403560:	74 0d                	je     0x40356f
  403562:	8b 4f 04             	mov    0x4(%edi),%ecx
  403565:	03 ce                	add    %esi,%ecx
  403567:	33 0c 30             	xor    (%eax,%esi,1),%ecx
  40356a:	e8 a2 fe ff ff       	call   0x403411
  40356f:	8b 57 08             	mov    0x8(%edi),%edx
  403572:	8b 4f 0c             	mov    0xc(%edi),%ecx
  403575:	03 ce                	add    %esi,%ecx
  403577:	33 0c 32             	xor    (%edx,%esi,1),%ecx
  40357a:	e8 92 fe ff ff       	call   0x403411
  40357f:	8b 45 f4             	mov    -0xc(%ebp),%eax
  403582:	5f                   	pop    %edi
  403583:	5e                   	pop    %esi
  403584:	5b                   	pop    %ebx
  403585:	8b e5                	mov    %ebp,%esp
  403587:	5d                   	pop    %ebp
  403588:	c3                   	ret
  403589:	8b 4f 04             	mov    0x4(%edi),%ecx
  40358c:	03 ce                	add    %esi,%ecx
  40358e:	33 0c 30             	xor    (%eax,%esi,1),%ecx
  403591:	e8 7b fe ff ff       	call   0x403411
  403596:	8b 47 08             	mov    0x8(%edi),%eax
  403599:	8b 4f 0c             	mov    0xc(%edi),%ecx
  40359c:	03 ce                	add    %esi,%ecx
  40359e:	33 0c 30             	xor    (%eax,%esi,1),%ecx
  4035a1:	e8 6b fe ff ff       	call   0x403411
  4035a6:	8b 4d f0             	mov    -0x10(%ebp),%ecx
  4035a9:	8b d6                	mov    %esi,%edx
  4035ab:	8b 49 08             	mov    0x8(%ecx),%ecx
  4035ae:	e8 46 02 00 00       	call   0x4037f9
  4035b3:	cc                   	int3
  4035b4:	55                   	push   %ebp
  4035b5:	8b ec                	mov    %esp,%ebp
  4035b7:	ff 15 a4 60 40 00    	call   *0x4060a4
  4035bd:	6a 01                	push   $0x1
  4035bf:	a3 e4 af 41 00       	mov    %eax,0x41afe4
  4035c4:	e8 79 02 00 00       	call   0x403842
  4035c9:	ff 75 08             	push   0x8(%ebp)
  4035cc:	e8 8e 02 00 00       	call   0x40385f
  4035d1:	83 3d e4 af 41 00 00 	cmpl   $0x0,0x41afe4
  4035d8:	59                   	pop    %ecx
  4035d9:	59                   	pop    %ecx
  4035da:	75 08                	jne    0x4035e4
  4035dc:	6a 01                	push   $0x1
  4035de:	e8 5f 02 00 00       	call   0x403842
  4035e3:	59                   	pop    %ecx
  4035e4:	68 09 04 00 c0       	push   $0xc0000409
  4035e9:	e8 5c 02 00 00       	call   0x40384a
  4035ee:	59                   	pop    %ecx
  4035ef:	5d                   	pop    %ebp
  4035f0:	c3                   	ret
  4035f1:	55                   	push   %ebp
  4035f2:	8b ec                	mov    %esp,%ebp
  4035f4:	81 ec 24 03 00 00    	sub    $0x324,%esp
  4035fa:	6a 17                	push   $0x17
  4035fc:	e8 7b 02 00 00       	call   0x40387c
  403601:	85 c0                	test   %eax,%eax
  403603:	74 05                	je     0x40360a
  403605:	6a 02                	push   $0x2
  403607:	59                   	pop    %ecx
  403608:	cd 29                	int    $0x29
  40360a:	a3 c8 ad 41 00       	mov    %eax,0x41adc8
  40360f:	89 0d c4 ad 41 00    	mov    %ecx,0x41adc4
  403615:	89 15 c0 ad 41 00    	mov    %edx,0x41adc0
  40361b:	89 1d bc ad 41 00    	mov    %ebx,0x41adbc
  403621:	89 35 b8 ad 41 00    	mov    %esi,0x41adb8
  403627:	89 3d b4 ad 41 00    	mov    %edi,0x41adb4
  40362d:	66 8c 15 e0 ad 41 00 	data16 mov %ss,0x41ade0
  403634:	66 8c 0d d4 ad 41 00 	data16 mov %cs,0x41add4
  40363b:	66 8c 1d b0 ad 41 00 	data16 mov %ds,0x41adb0
  403642:	66 8c 05 ac ad 41 00 	data16 mov %es,0x41adac
  403649:	66 8c 25 a8 ad 41 00 	data16 mov %fs,0x41ada8
  403650:	66 8c 2d a4 ad 41 00 	data16 mov %gs,0x41ada4
  403657:	9c                   	pushf
  403658:	8f 05 d8 ad 41 00    	pop    0x41add8
  40365e:	8b 45 00             	mov    0x0(%ebp),%eax
  403661:	a3 cc ad 41 00       	mov    %eax,0x41adcc
  403666:	8b 45 04             	mov    0x4(%ebp),%eax
  403669:	a3 d0 ad 41 00       	mov    %eax,0x41add0
  40366e:	8d 45 08             	lea    0x8(%ebp),%eax
  403671:	a3 dc ad 41 00       	mov    %eax,0x41addc
  403676:	8b 85 dc fc ff ff    	mov    -0x324(%ebp),%eax
  40367c:	c7 05 18 ad 41 00 01 	movl   $0x10001,0x41ad18
  403683:	00 01 00 
  403686:	a1 d0 ad 41 00       	mov    0x41add0,%eax
  40368b:	a3 d4 ac 41 00       	mov    %eax,0x41acd4
  403690:	c7 05 c8 ac 41 00 09 	movl   $0xc0000409,0x41acc8
  403697:	04 00 c0 
  40369a:	c7 05 cc ac 41 00 01 	movl   $0x1,0x41accc
  4036a1:	00 00 00 
  4036a4:	c7 05 d8 ac 41 00 01 	movl   $0x1,0x41acd8
  4036ab:	00 00 00 
  4036ae:	6a 04                	push   $0x4
  4036b0:	58                   	pop    %eax
  4036b1:	6b c0 00             	imul   $0x0,%eax,%eax
  4036b4:	c7 80 dc ac 41 00 02 	movl   $0x2,0x41acdc(%eax)
  4036bb:	00 00 00 
  4036be:	6a 04                	push   $0x4
  4036c0:	58                   	pop    %eax
  4036c1:	6b c0 00             	imul   $0x0,%eax,%eax
  4036c4:	8b 0d d0 a7 41 00    	mov    0x41a7d0,%ecx
  4036ca:	89 4c 05 f8          	mov    %ecx,-0x8(%ebp,%eax,1)
  4036ce:	6a 04                	push   $0x4
  4036d0:	58                   	pop    %eax
  4036d1:	c1 e0 00             	shl    $0x0,%eax
  4036d4:	8b 0d d4 a7 41 00    	mov    0x41a7d4,%ecx
  4036da:	89 4c 05 f8          	mov    %ecx,-0x8(%ebp,%eax,1)
  4036de:	68 60 72 40 00       	push   $0x407260
  4036e3:	e8 cc fe ff ff       	call   0x4035b4
  4036e8:	8b e5                	mov    %ebp,%esp
  4036ea:	5d                   	pop    %ebp
  4036eb:	c3                   	ret
  4036ec:	cc                   	int3
  4036ed:	cc                   	int3
  4036ee:	cc                   	int3
  4036ef:	cc                   	int3
  4036f0:	53                   	push   %ebx
  4036f1:	56                   	push   %esi
  4036f2:	57                   	push   %edi
  4036f3:	8b 54 24 10          	mov    0x10(%esp),%edx
  4036f7:	8b 44 24 14          	mov    0x14(%esp),%eax
  4036fb:	8b 4c 24 18          	mov    0x18(%esp),%ecx
  4036ff:	55                   	push   %ebp
  403700:	52                   	push   %edx
  403701:	50                   	push   %eax
  403702:	51                   	push   %ecx
  403703:	51                   	push   %ecx
  403704:	68 80 37 40 00       	push   $0x403780
  403709:	64 ff 35 00 00 00 00 	push   %fs:0x0
  403710:	a1 d0 a7 41 00       	mov    0x41a7d0,%eax
  403715:	33 c4                	xor    %esp,%eax
  403717:	89 44 24 08          	mov    %eax,0x8(%esp)
  40371b:	64 89 25 00 00 00 00 	mov    %esp,%fs:0x0
  403722:	8b 44 24 30          	mov    0x30(%esp),%eax
  403726:	8b 58 08             	mov    0x8(%eax),%ebx
  403729:	8b 4c 24 2c          	mov    0x2c(%esp),%ecx
  40372d:	33 19                	xor    (%ecx),%ebx
  40372f:	8b 70 0c             	mov    0xc(%eax),%esi
  403732:	83 fe fe             	cmp    $0xfffffffe,%esi
  403735:	74 3b                	je     0x403772
  403737:	8b 54 24 34          	mov    0x34(%esp),%edx
  40373b:	83 fa fe             	cmp    $0xfffffffe,%edx
  40373e:	74 04                	je     0x403744
  403740:	3b f2                	cmp    %edx,%esi
  403742:	76 2e                	jbe    0x403772
  403744:	8d 34 76             	lea    (%esi,%esi,2),%esi
  403747:	8d 5c b3 10          	lea    0x10(%ebx,%esi,4),%ebx
  40374b:	8b 0b                	mov    (%ebx),%ecx
  40374d:	89 48 0c             	mov    %ecx,0xc(%eax)
  403750:	83 7b 04 00          	cmpl   $0x0,0x4(%ebx)
  403754:	75 cc                	jne    0x403722
  403756:	68 01 01 00 00       	push   $0x101
  40375b:	8b 43 08             	mov    0x8(%ebx),%eax
  40375e:	e8 12 f7 ff ff       	call   0x402e75
  403763:	b9 01 00 00 00       	mov    $0x1,%ecx
  403768:	8b 43 08             	mov    0x8(%ebx),%eax
  40376b:	e8 24 f7 ff ff       	call   0x402e94
  403770:	eb b0                	jmp    0x403722
  403772:	64 8f 05 00 00 00 00 	pop    %fs:0x0
  403779:	83 c4 18             	add    $0x18,%esp
  40377c:	5f                   	pop    %edi
  40377d:	5e                   	pop    %esi
  40377e:	5b                   	pop    %ebx
  40377f:	c3                   	ret
  403780:	8b 4c 24 04          	mov    0x4(%esp),%ecx
  403784:	f7 41 04 06 00 00 00 	testl  $0x6,0x4(%ecx)
  40378b:	b8 01 00 00 00       	mov    $0x1,%eax
  403790:	74 33                	je     0x4037c5
  403792:	8b 44 24 08          	mov    0x8(%esp),%eax
  403796:	8b 48 08             	mov    0x8(%eax),%ecx
  403799:	33 c8                	xor    %eax,%ecx
  40379b:	e8 71 fc ff ff       	call   0x403411
  4037a0:	55                   	push   %ebp
  4037a1:	8b 68 18             	mov    0x18(%eax),%ebp
  4037a4:	ff 70 0c             	push   0xc(%eax)
  4037a7:	ff 70 10             	push   0x10(%eax)
  4037aa:	ff 70 14             	push   0x14(%eax)
  4037ad:	e8 3e ff ff ff       	call   0x4036f0
  4037b2:	83 c4 0c             	add    $0xc,%esp
  4037b5:	5d                   	pop    %ebp
  4037b6:	8b 44 24 08          	mov    0x8(%esp),%eax
  4037ba:	8b 54 24 10          	mov    0x10(%esp),%edx
  4037be:	89 02                	mov    %eax,(%edx)
  4037c0:	b8 03 00 00 00       	mov    $0x3,%eax
  4037c5:	c3                   	ret
  4037c6:	55                   	push   %ebp
  4037c7:	8b 4c 24 08          	mov    0x8(%esp),%ecx
  4037cb:	8b 29                	mov    (%ecx),%ebp
  4037cd:	ff 71 1c             	push   0x1c(%ecx)
  4037d0:	ff 71 18             	push   0x18(%ecx)
  4037d3:	ff 71 28             	push   0x28(%ecx)
  4037d6:	e8 15 ff ff ff       	call   0x4036f0
  4037db:	83 c4 0c             	add    $0xc,%esp
  4037de:	5d                   	pop    %ebp
  4037df:	c2 04 00             	ret    $0x4
  4037e2:	55                   	push   %ebp
  4037e3:	56                   	push   %esi
  4037e4:	57                   	push   %edi
  4037e5:	53                   	push   %ebx
  4037e6:	8b ea                	mov    %edx,%ebp
  4037e8:	33 c0                	xor    %eax,%eax
  4037ea:	33 db                	xor    %ebx,%ebx
  4037ec:	33 d2                	xor    %edx,%edx
  4037ee:	33 f6                	xor    %esi,%esi
  4037f0:	33 ff                	xor    %edi,%edi
  4037f2:	ff d1                	call   *%ecx
  4037f4:	5b                   	pop    %ebx
  4037f5:	5f                   	pop    %edi
  4037f6:	5e                   	pop    %esi
  4037f7:	5d                   	pop    %ebp
  4037f8:	c3                   	ret
  4037f9:	8b ea                	mov    %edx,%ebp
  4037fb:	8b f1                	mov    %ecx,%esi
  4037fd:	8b c1                	mov    %ecx,%eax
  4037ff:	6a 01                	push   $0x1
  403801:	e8 6f f6 ff ff       	call   0x402e75
  403806:	33 c0                	xor    %eax,%eax
  403808:	33 db                	xor    %ebx,%ebx
  40380a:	33 c9                	xor    %ecx,%ecx
  40380c:	33 d2                	xor    %edx,%edx
  40380e:	33 ff                	xor    %edi,%edi
  403810:	ff e6                	jmp    *%esi
  403812:	55                   	push   %ebp
  403813:	8b ec                	mov    %esp,%ebp
  403815:	53                   	push   %ebx
  403816:	56                   	push   %esi
  403817:	57                   	push   %edi
  403818:	6a 00                	push   $0x0
  40381a:	52                   	push   %edx
  40381b:	68 26 38 40 00       	push   $0x403826
  403820:	51                   	push   %ecx
  403821:	e8 50 00 00 00       	call   0x403876
  403826:	5f                   	pop    %edi
  403827:	5e                   	pop    %esi
  403828:	5b                   	pop    %ebx
  403829:	5d                   	pop    %ebp
  40382a:	c3                   	ret
  40382b:	55                   	push   %ebp
  40382c:	8b 6c 24 08          	mov    0x8(%esp),%ebp
  403830:	52                   	push   %edx
  403831:	51                   	push   %ecx
  403832:	ff 74 24 14          	push   0x14(%esp)
  403836:	e8 b5 fe ff ff       	call   0x4036f0
  40383b:	83 c4 0c             	add    $0xc,%esp
  40383e:	5d                   	pop    %ebp
  40383f:	c2 08 00             	ret    $0x8
  403842:	83 25 e8 af 41 00 00 	andl   $0x0,0x41afe8
  403849:	c3                   	ret
  40384a:	55                   	push   %ebp
  40384b:	8b ec                	mov    %esp,%ebp
  40384d:	ff 75 08             	push   0x8(%ebp)
  403850:	ff 15 54 60 40 00    	call   *0x406054
  403856:	50                   	push   %eax
  403857:	ff 15 40 60 40 00    	call   *0x406040
  40385d:	5d                   	pop    %ebp
  40385e:	c3                   	ret
  40385f:	55                   	push   %ebp
  403860:	8b ec                	mov    %esp,%ebp
  403862:	6a 00                	push   $0x0
  403864:	ff 15 30 60 40 00    	call   *0x406030
  40386a:	ff 75 08             	push   0x8(%ebp)
  40386d:	ff 15 9c 60 40 00    	call   *0x40609c
  403873:	5d                   	pop    %ebp
  403874:	c3                   	ret
  403875:	cc                   	int3
  403876:	ff 25 a8 60 40 00    	jmp    *0x4060a8
  40387c:	ff 25 a0 60 40 00    	jmp    *0x4060a0
  403882:	cc                   	int3
  403883:	cc                   	int3
  403884:	cc                   	int3
  403885:	cc                   	int3
  403886:	cc                   	int3
  403887:	cc                   	int3
  403888:	cc                   	int3
  403889:	cc                   	int3
  40388a:	cc                   	int3
  40388b:	cc                   	int3
  40388c:	cc                   	int3
  40388d:	cc                   	int3
  40388e:	cc                   	int3
  40388f:	cc                   	int3
  403890:	55                   	push   %ebp
  403891:	83 ec 27             	sub    $0x27,%esp
  403894:	89 e5                	mov    %esp,%ebp
  403896:	51                   	push   %ecx
  403897:	52                   	push   %edx
  403898:	56                   	push   %esi
  403899:	e8 80 10 00 00       	call   0x40491e
  40389e:	60                   	pusha
  40389f:	11 00                	adc    %eax,(%eax)
  4038a1:	00 5c 11 00          	add    %bl,0x0(%ecx,%edx,1)
  4038a5:	00 58 11             	add    %bl,0x11(%eax)
  4038a8:	00 00                	add    %al,(%eax)
  4038aa:	54                   	push   %esp
  4038ab:	11 00                	adc    %eax,(%eax)
  4038ad:	00 64 11 00          	add    %ah,0x0(%ecx,%edx,1)
  4038b1:	00 9c 11 00 00 88 11 	add    %bl,0x11880000(%ecx,%edx,1)
  4038b8:	00 00                	add    %al,(%eax)
  4038ba:	84 11                	test   %dl,(%ecx)
  4038bc:	00 00                	add    %al,(%eax)
  4038be:	40                   	inc    %eax
  4038bf:	11 00                	adc    %eax,(%eax)
  4038c1:	00 3c 11             	add    %bh,(%ecx,%edx,1)
  4038c4:	00 00                	add    %al,(%eax)
  4038c6:	38 11                	cmp    %dl,(%ecx)
  4038c8:	00 00                	add    %al,(%eax)
  4038ca:	34 11                	xor    $0x11,%al
  4038cc:	00 00                	add    %al,(%eax)
  4038ce:	44                   	inc    %esp
  4038cf:	11 00                	adc    %eax,(%eax)
  4038d1:	00 7c 11 00          	add    %bh,0x0(%ecx,%edx,1)
  4038d5:	00 68 11             	add    %ch,0x11(%eax)
  4038d8:	00 00                	add    %al,(%eax)
  4038da:	39 1a                	cmp    %ebx,(%edx)
  4038dc:	00 00                	add    %al,(%eax)
  4038de:	20 11                	and    %dl,(%ecx)
  4038e0:	00 00                	add    %al,(%eax)
  4038e2:	1c 11                	sbb    $0x11,%al
  4038e4:	00 00                	add    %al,(%eax)
  4038e6:	18 11                	sbb    %dl,(%ecx)
  4038e8:	00 00                	add    %al,(%eax)
  4038ea:	14 11                	adc    $0x11,%al
  4038ec:	00 00                	add    %al,(%eax)
  4038ee:	24 11                	and    $0x11,%al
  4038f0:	00 00                	add    %al,(%eax)
  4038f2:	5c                   	pop    %esp
  4038f3:	11 00                	adc    %eax,(%eax)
  4038f5:	00 48 11             	add    %cl,0x11(%eax)
  4038f8:	00 00                	add    %al,(%eax)
  4038fa:	44                   	inc    %esp
  4038fb:	11 00                	adc    %eax,(%eax)
  4038fd:	00 00                	add    %al,(%eax)
  4038ff:	11 00                	adc    %eax,(%eax)
  403901:	00 fc                	add    %bh,%ah
  403903:	10 00                	adc    %al,(%eax)
  403905:	00 f8                	add    %bh,%al
  403907:	10 00                	adc    %al,(%eax)
  403909:	00 f4                	add    %dh,%ah
  40390b:	10 00                	adc    %al,(%eax)
  40390d:	00 04 11             	add    %al,(%ecx,%edx,1)
  403910:	00 00                	add    %al,(%eax)
  403912:	3c 11                	cmp    $0x11,%al
  403914:	00 00                	add    %al,(%eax)
  403916:	28 11                	sub    %dl,(%ecx)
  403918:	00 00                	add    %al,(%eax)
  40391a:	24 11                	and    $0x11,%al
  40391c:	00 00                	add    %al,(%eax)
  40391e:	e0 10                	loopne 0x403930
  403920:	00 00                	add    %al,(%eax)
  403922:	dc 10                	fcoml  (%eax)
  403924:	00 00                	add    %al,(%eax)
  403926:	d8 10                	fcoms  (%eax)
  403928:	00 00                	add    %al,(%eax)
  40392a:	d4 10                	aam    $0x10
  40392c:	00 00                	add    %al,(%eax)
  40392e:	e4 10                	in     $0x10,%al
  403930:	00 00                	add    %al,(%eax)
  403932:	1c 11                	sbb    $0x11,%al
  403934:	00 00                	add    %al,(%eax)
  403936:	a6                   	cmpsb  %es:(%edi),%ds:(%esi)
  403937:	11 00                	adc    %eax,(%eax)
  403939:	00 04 11             	add    %al,(%ecx,%edx,1)
  40393c:	00 00                	add    %al,(%eax)
  40393e:	c0 10 00             	rclb   $0x0,(%eax)
  403941:	00 bc 10 00 00 b8 10 	add    %bh,0x10b80000(%eax,%edx,1)
  403948:	00 00                	add    %al,(%eax)
  40394a:	b4 10                	mov    $0x10,%ah
  40394c:	00 00                	add    %al,(%eax)
  40394e:	c4 10                	les    (%eax),%edx
  403950:	00 00                	add    %al,(%eax)
  403952:	fc                   	cld
  403953:	10 00                	adc    %al,(%eax)
  403955:	00 86 11 00 00 e4    	add    %al,-0x1bffffef(%esi)
  40395b:	10 00                	adc    %al,(%eax)
  40395d:	00 a0 10 00 00 9c    	add    %ah,-0x63fffff0(%eax)
  403963:	10 00                	adc    %al,(%eax)
  403965:	00 98 10 00 00 94    	add    %bl,-0x6bfffff0(%eax)
  40396b:	10 00                	adc    %al,(%eax)
  40396d:	00 a4 10 00 00 dc 10 	add    %ah,0x10dc0000(%eax,%edx,1)
  403974:	00 00                	add    %al,(%eax)
  403976:	66 11 00             	adc    %ax,(%eax)
  403979:	00 c4                	add    %al,%ah
  40397b:	10 00                	adc    %al,(%eax)
  40397d:	00 80 10 00 00 7c    	add    %al,0x7c000010(%eax)
  403983:	10 00                	adc    %al,(%eax)
  403985:	00 78 10             	add    %bh,0x10(%eax)
  403988:	00 00                	add    %al,(%eax)
  40398a:	74 10                	je     0x40399c
  40398c:	00 00                	add    %al,(%eax)
  40398e:	84 10                	test   %dl,(%eax)
  403990:	00 00                	add    %al,(%eax)
  403992:	bc 10 00 00 46       	mov    $0x46000010,%esp
  403997:	11 00                	adc    %eax,(%eax)
  403999:	00 a4 10 00 00 10 11 	add    %ah,0x11100000(%eax,%edx,1)
  4039a0:	00 00                	add    %al,(%eax)
  4039a2:	0c 11                	or     $0x11,%al
  4039a4:	00 00                	add    %al,(%eax)
  4039a6:	08 11                	or     %dl,(%ecx)
  4039a8:	00 00                	add    %al,(%eax)
  4039aa:	04 11                	add    $0x11,%al
  4039ac:	00 00                	add    %al,(%eax)
  4039ae:	00 11                	add    %dl,(%ecx)
  4039b0:	00 00                	add    %al,(%eax)
  4039b2:	fc                   	cld
  4039b3:	10 00                	adc    %al,(%eax)
  4039b5:	00 f8                	add    %bh,%al
  4039b7:	10 00                	adc    %al,(%eax)
  4039b9:	00 f4                	add    %dh,%ah
  4039bb:	10 00                	adc    %al,(%eax)
  4039bd:	00 c4                	add    %al,%ah
  4039bf:	10 00                	adc    %al,(%eax)
  4039c1:	00 c0                	add    %al,%al
  4039c3:	10 00                	adc    %al,(%eax)
  4039c5:	00 bc 10 00 00 b8 10 	add    %bh,0x10b80000(%eax,%edx,1)
  4039cc:	00 00                	add    %al,(%eax)
  4039ce:	b4 10                	mov    $0x10,%ah
  4039d0:	00 00                	add    %al,(%eax)
  4039d2:	b0 10                	mov    $0x10,%al
  4039d4:	00 00                	add    %al,(%eax)
  4039d6:	ac                   	lods   %ds:(%esi),%al
  4039d7:	10 00                	adc    %al,(%eax)
  4039d9:	00 a8 10 00 00 30    	add    %ch,0x30000010(%eax)
  4039df:	10 00                	adc    %al,(%eax)
  4039e1:	00 2c 10             	add    %ch,(%eax,%edx,1)
  4039e4:	00 00                	add    %al,(%eax)
  4039e6:	28 10                	sub    %dl,(%eax)
  4039e8:	00 00                	add    %al,(%eax)
  4039ea:	24 10                	and    $0x10,%al
  4039ec:	00 00                	add    %al,(%eax)
  4039ee:	20 10                	and    %dl,(%eax)
  4039f0:	00 00                	add    %al,(%eax)
  4039f2:	1c 10                	sbb    $0x10,%al
  4039f4:	00 00                	add    %al,(%eax)
  4039f6:	18 10                	sbb    %dl,(%eax)
  4039f8:	00 00                	add    %al,(%eax)
  4039fa:	14 10                	adc    $0x10,%al
  4039fc:	00 00                	add    %al,(%eax)
  4039fe:	10 10                	adc    %dl,(%eax)
  403a00:	00 00                	add    %al,(%eax)
  403a02:	0c 10                	or     $0x10,%al
  403a04:	00 00                	add    %al,(%eax)
  403a06:	08 10                	or     %dl,(%eax)
  403a08:	00 00                	add    %al,(%eax)
  403a0a:	04 10                	add    $0x10,%al
  403a0c:	00 00                	add    %al,(%eax)
  403a0e:	00 10                	add    %dl,(%eax)
  403a10:	00 00                	add    %al,(%eax)
  403a12:	fc                   	cld
  403a13:	0f 00 00             	sldt   (%eax)
  403a16:	f8                   	clc
  403a17:	0f 00 00             	sldt   (%eax)
  403a1a:	f4                   	hlt
  403a1b:	0f 00 00             	sldt   (%eax)
  403a1e:	f0 0f 00 00          	lock sldt (%eax)
  403a22:	ec                   	in     (%dx),%al
  403a23:	0f 00 00             	sldt   (%eax)
  403a26:	0e                   	push   %cs
  403a27:	11 00                	adc    %eax,(%eax)
  403a29:	00 d4                	add    %dl,%ah
  403a2b:	0f 00 00             	sldt   (%eax)
  403a2e:	ae                   	scas   %es:(%edi),%al
  403a2f:	10 00                	adc    %al,(%eax)
  403a31:	00 aa 10 00 00 bf    	add    %ch,-0x40fffff0(%edx)
  403a37:	17                   	pop    %ss
  403a38:	00 00                	add    %al,(%eax)
  403a3a:	ec                   	in     (%dx),%al
  403a3b:	17                   	pop    %ss
  403a3c:	00 00                	add    %al,(%eax)
  403a3e:	01 12                	add    %edx,(%edx)
  403a40:	00 00                	add    %al,(%eax)
  403a42:	a4                   	movsb  %ds:(%esi),%es:(%edi)
  403a43:	11 00                	adc    %eax,(%eax)
  403a45:	00 cc                	add    %cl,%ah
  403a47:	0f 00 00             	sldt   (%eax)
  403a4a:	2f                   	das
  403a4b:	10 00                	adc    %al,(%eax)
  403a4d:	00 c0                	add    %al,%al
  403a4f:	0f 00 00             	sldt   (%eax)
  403a52:	bc 0f 00 00 b8       	mov    $0xb800000f,%esp
  403a57:	0f 00 00             	sldt   (%eax)
  403a5a:	b4 0f                	mov    $0xf,%ah
  403a5c:	00 00                	add    %al,(%eax)
  403a5e:	b4 0f                	mov    $0xf,%ah
  403a60:	00 00                	add    %al,(%eax)
  403a62:	b0 0f                	mov    $0xf,%al
  403a64:	00 00                	add    %al,(%eax)
  403a66:	ac                   	lods   %ds:(%esi),%al
  403a67:	0f 00 00             	sldt   (%eax)
  403a6a:	a8 0f                	test   $0xf,%al
  403a6c:	00 00                	add    %al,(%eax)
  403a6e:	a4                   	movsb  %ds:(%esi),%es:(%edi)
  403a6f:	0f 00 00             	sldt   (%eax)
  403a72:	a0 0f 00 00 9c       	mov    0x9c00000f,%al
  403a77:	0f 00 00             	sldt   (%eax)
  403a7a:	98                   	cwtl
  403a7b:	0f 00 00             	sldt   (%eax)
  403a7e:	94                   	xchg   %eax,%esp
  403a7f:	0f 00 00             	sldt   (%eax)
  403a82:	90                   	nop
  403a83:	0f 00 00             	sldt   (%eax)
  403a86:	8c 0f                	mov    %cs,(%edi)
  403a88:	00 00                	add    %al,(%eax)
  403a8a:	88 0f                	mov    %cl,(%edi)
  403a8c:	00 00                	add    %al,(%eax)
  403a8e:	84 0f                	test   %cl,(%edi)
  403a90:	00 00                	add    %al,(%eax)
  403a92:	80 0f 00             	orb    $0x0,(%edi)
  403a95:	00 7c 0f 00          	add    %bh,0x0(%edi,%ecx,1)
  403a99:	00 78 0f             	add    %bh,0xf(%eax)
  403a9c:	00 00                	add    %al,(%eax)
  403a9e:	db 0f                	fisttpl (%edi)
  403aa0:	00 00                	add    %al,(%eax)
  403aa2:	e6 11                	out    %al,$0x11
  403aa4:	00 00                	add    %al,(%eax)
  403aa6:	cc                   	int3
  403aa7:	11 00                	adc    %eax,(%eax)
  403aa9:	00 cf                	add    %cl,%bh
  403aab:	0f 00 00             	sldt   (%eax)
  403aae:	50                   	push   %eax
  403aaf:	0f 00 00             	sldt   (%eax)
  403ab2:	4c                   	dec    %esp
  403ab3:	0f 00 00             	sldt   (%eax)
  403ab6:	48                   	dec    %eax
  403ab7:	0f 00 00             	sldt   (%eax)
  403aba:	44                   	inc    %esp
  403abb:	0f 00 00             	sldt   (%eax)
  403abe:	40                   	inc    %eax
  403abf:	0f 00 00             	sldt   (%eax)
  403ac2:	3c 0f                	cmp    $0xf,%al
  403ac4:	00 00                	add    %al,(%eax)
  403ac6:	38 0f                	cmp    %cl,(%edi)
  403ac8:	00 00                	add    %al,(%eax)
  403aca:	34 0f                	xor    $0xf,%al
  403acc:	00 00                	add    %al,(%eax)
  403ace:	30 0f                	xor    %cl,(%edi)
  403ad0:	00 00                	add    %al,(%eax)
  403ad2:	2c 0f                	sub    $0xf,%al
  403ad4:	00 00                	add    %al,(%eax)
  403ad6:	28 0f                	sub    %cl,(%edi)
  403ad8:	00 00                	add    %al,(%eax)
  403ada:	4e                   	dec    %esi
  403adb:	11 00                	adc    %eax,(%eax)
  403add:	00 30                	add    %dh,(%eax)
  403adf:	0f 00 00             	sldt   (%eax)
  403ae2:	2c 0f                	sub    $0xf,%al
  403ae4:	00 00                	add    %al,(%eax)
  403ae6:	28 0f                	sub    %cl,(%edi)
  403ae8:	00 00                	add    %al,(%eax)
  403aea:	24 0f                	and    $0xf,%al
  403aec:	00 00                	add    %al,(%eax)
  403aee:	20 0f                	and    %cl,(%edi)
  403af0:	00 00                	add    %al,(%eax)
  403af2:	1c 0f                	sbb    $0xf,%al
  403af4:	00 00                	add    %al,(%eax)
  403af6:	18 0f                	sbb    %cl,(%edi)
  403af8:	00 00                	add    %al,(%eax)
  403afa:	14 0f                	adc    $0xf,%al
  403afc:	00 00                	add    %al,(%eax)
  403afe:	10 0f                	adc    %cl,(%edi)
  403b00:	00 00                	add    %al,(%eax)
  403b02:	0c 0f                	or     $0xf,%al
  403b04:	00 00                	add    %al,(%eax)
  403b06:	c3                   	ret
  403b07:	10 00                	adc    %al,(%eax)
  403b09:	00 04 0f             	add    %al,(%edi,%ecx,1)
  403b0c:	00 00                	add    %al,(%eax)
  403b0e:	00 0f                	add    %cl,(%edi)
  403b10:	00 00                	add    %al,(%eax)
  403b12:	fc                   	cld
  403b13:	0e                   	push   %cs
  403b14:	00 00                	add    %al,(%eax)
  403b16:	f8                   	clc
  403b17:	0e                   	push   %cs
  403b18:	00 00                	add    %al,(%eax)
  403b1a:	f4                   	hlt
  403b1b:	0e                   	push   %cs
  403b1c:	00 00                	add    %al,(%eax)
  403b1e:	44                   	inc    %esp
  403b1f:	10 00                	adc    %al,(%eax)
  403b21:	00 50 10             	add    %dl,0x10(%eax)
  403b24:	00 00                	add    %al,(%eax)
  403b26:	3c 10                	cmp    $0x10,%al
  403b28:	00 00                	add    %al,(%eax)
  403b2a:	48                   	dec    %eax
  403b2b:	10 00                	adc    %al,(%eax)
  403b2d:	00 e0                	add    %ah,%al
  403b2f:	0e                   	push   %cs
  403b30:	00 00                	add    %al,(%eax)
  403b32:	dc 0e                	fmull  (%esi)
  403b34:	00 00                	add    %al,(%eax)
  403b36:	d8 0e                	fmuls  (%esi)
  403b38:	00 00                	add    %al,(%eax)
  403b3a:	d4 0e                	aam    $0xe
  403b3c:	00 00                	add    %al,(%eax)
  403b3e:	d4 0e                	aam    $0xe
  403b40:	00 00                	add    %al,(%eax)
  403b42:	0c 0f                	or     $0xf,%al
  403b44:	00 00                	add    %al,(%eax)
  403b46:	c8 0e 00 00          	enter  $0xe,$0x0
  403b4a:	c4 0e                	les    (%esi),%ecx
  403b4c:	00 00                	add    %al,(%eax)
  403b4e:	c0 0e 00             	rorb   $0x0,(%esi)
  403b51:	00 bc 0e 00 00 b8 0e 	add    %bh,0xeb80000(%esi,%ecx,1)
  403b58:	00 00                	add    %al,(%eax)
  403b5a:	b4 0e                	mov    $0xe,%ah
  403b5c:	00 00                	add    %al,(%eax)
  403b5e:	b4 0e                	mov    $0xe,%ah
  403b60:	00 00                	add    %al,(%eax)
  403b62:	b0 0e                	mov    $0xe,%al
  403b64:	00 00                	add    %al,(%eax)
  403b66:	ac                   	lods   %ds:(%esi),%al
  403b67:	0e                   	push   %cs
  403b68:	00 00                	add    %al,(%eax)
  403b6a:	a8 0e                	test   $0xe,%al
  403b6c:	00 00                	add    %al,(%eax)
  403b6e:	a4                   	movsb  %ds:(%esi),%es:(%edi)
  403b6f:	0e                   	push   %cs
  403b70:	00 00                	add    %al,(%eax)
  403b72:	a0 0e 00 00 9c       	mov    0x9c00000e,%al
  403b77:	0e                   	push   %cs
  403b78:	00 00                	add    %al,(%eax)
  403b7a:	98                   	cwtl
  403b7b:	0e                   	push   %cs
  403b7c:	00 00                	add    %al,(%eax)
  403b7e:	e0 0e                	loopne 0x403b8e
  403b80:	00 00                	add    %al,(%eax)
  403b82:	dc 0e                	fmull  (%esi)
  403b84:	00 00                	add    %al,(%eax)
  403b86:	d8 0e                	fmuls  (%esi)
  403b88:	00 00                	add    %al,(%eax)
  403b8a:	d4 0e                	aam    $0xe
  403b8c:	00 00                	add    %al,(%eax)
  403b8e:	d0 0e                	rorb   $1,(%esi)
  403b90:	00 00                	add    %al,(%eax)
  403b92:	cc                   	int3
  403b93:	0e                   	push   %cs
  403b94:	00 00                	add    %al,(%eax)
  403b96:	c8 0e 00 00          	enter  $0xe,$0x0
  403b9a:	c4 0e                	les    (%esi),%ecx
  403b9c:	00 00                	add    %al,(%eax)
  403b9e:	db 0e                	fisttpl (%esi)
  403ba0:	00 00                	add    %al,(%eax)
  403ba2:	d7                   	xlat   %ds:(%ebx)
  403ba3:	0e                   	push   %cs
  403ba4:	00 00                	add    %al,(%eax)
  403ba6:	b7 0f                	mov    $0xf,%bh
  403ba8:	00 00                	add    %al,(%eax)
  403baa:	64 0e                	fs push %cs
  403bac:	00 00                	add    %al,(%eax)
  403bae:	86 0f                	xchg   %cl,(%edi)
  403bb0:	00 00                	add    %al,(%eax)
  403bb2:	82 0f 00             	orb    $0x0,(%edi)
  403bb5:	00 c3                	add    %al,%bl
  403bb7:	0e                   	push   %cs
  403bb8:	00 00                	add    %al,(%eax)
  403bba:	45                   	inc    %ebp
  403bbb:	0f 00 00             	sldt   (%eax)
  403bbe:	6c                   	insb   (%dx),%es:(%edi)
  403bbf:	0f 00 00             	sldt   (%eax)
  403bc2:	4c                   	dec    %esp
  403bc3:	0e                   	push   %cs
  403bc4:	00 00                	add    %al,(%eax)
  403bc6:	97                   	xchg   %eax,%edi
  403bc7:	0f 00 00             	sldt   (%eax)
  403bca:	44                   	inc    %esp
  403bcb:	0e                   	push   %cs
  403bcc:	00 00                	add    %al,(%eax)
  403bce:	40                   	inc    %eax
  403bcf:	0e                   	push   %cs
  403bd0:	00 00                	add    %al,(%eax)
  403bd2:	40                   	inc    %eax
  403bd3:	0e                   	push   %cs
  403bd4:	00 00                	add    %al,(%eax)
  403bd6:	68 0e 00 00 34       	push   $0x3400000e
  403bdb:	0e                   	push   %cs
  403bdc:	00 00                	add    %al,(%eax)
  403bde:	20 0e                	and    %cl,(%esi)
  403be0:	00 00                	add    %al,(%eax)
  403be2:	1c 0e                	sbb    $0xe,%al
  403be4:	00 00                	add    %al,(%eax)
  403be6:	18 0e                	sbb    %cl,(%esi)
  403be8:	00 00                	add    %al,(%eax)
  403bea:	14 0e                	adc    $0xe,%al
  403bec:	00 00                	add    %al,(%eax)
  403bee:	2b 0f                	sub    (%edi),%ecx
  403bf0:	00 00                	add    %al,(%eax)
  403bf2:	27                   	daa
  403bf3:	0f 00 00             	sldt   (%eax)
  403bf6:	18 0e                	sbb    %cl,(%esi)
  403bf8:	00 00                	add    %al,(%eax)
  403bfa:	14 0e                	adc    $0xe,%al
  403bfc:	00 00                	add    %al,(%eax)
  403bfe:	8a 17                	mov    (%edi),%dl
  403c00:	00 00                	add    %al,(%eax)
  403c02:	bc 17 00 00 51       	mov    $0x51000017,%esp
  403c07:	18 00                	sbb    %al,(%eax)
  403c09:	00 af 18 00 00 2a    	add    %ch,0x2a000018(%edi)
  403c0f:	19 00                	sbb    %eax,(%eax)
  403c11:	00 78 19             	add    %bh,0x19(%eax)
  403c14:	00 00                	add    %al,(%eax)
  403c16:	de 19                	ficomps (%ecx)
  403c18:	00 00                	add    %al,(%eax)
  403c1a:	31 1a                	xor    %ebx,(%edx)
  403c1c:	00 00                	add    %al,(%eax)
  403c1e:	f4                   	hlt
  403c1f:	0d 00 00 f0 0d       	or     $0xdf00000,%eax
  403c24:	00 00                	add    %al,(%eax)
  403c26:	ec                   	in     (%dx),%al
  403c27:	0d 00 00 e8 0d       	or     $0xde80000,%eax
  403c2c:	00 00                	add    %al,(%eax)
  403c2e:	e4 0d                	in     $0xd,%al
  403c30:	00 00                	add    %al,(%eax)
  403c32:	e0 0d                	loopne 0x403c41
  403c34:	00 00                	add    %al,(%eax)
  403c36:	dc 0d 00 00 d8 0d    	fmull  0xdd80000
  403c3c:	00 00                	add    %al,(%eax)
  403c3e:	10 0e                	adc    %cl,(%esi)
  403c40:	00 00                	add    %al,(%eax)
  403c42:	0c 0e                	or     $0xe,%al
  403c44:	00 00                	add    %al,(%eax)
  403c46:	c6                   	(bad)
  403c47:	0f 00 00             	sldt   (%eax)
  403c4a:	c8 0d 00 00          	enter  $0xd,$0x0
  403c4e:	c0 0d 00 00 bc 0d 00 	rorb   $0x0,0xdbc0000
  403c55:	00 b8 0d 00 00 b4    	add    %bh,-0x4bfffff3(%eax)
  403c5b:	0d 00 00 7e 0e       	or     $0xe7e0000,%eax
  403c60:	00 00                	add    %al,(%eax)
  403c62:	ac                   	lods   %ds:(%esi),%al
  403c63:	0d 00 00 f3 15       	or     $0x15f30000,%eax
  403c68:	00 00                	add    %al,(%eax)
  403c6a:	36 16                	ss push %ss
  403c6c:	00 00                	add    %al,(%eax)
  403c6e:	a0 0d 00 00 9c       	mov    0x9c00000d,%al
  403c73:	0d 00 00 2c 10       	or     $0x102c0000,%eax
  403c78:	00 00                	add    %al,(%eax)
  403c7a:	5a                   	pop    %edx
  403c7b:	10 00                	adc    %al,(%eax)
  403c7d:	00 90 0d 00 00 8c    	add    %dl,-0x73fffff3(%eax)
  403c83:	0d 00 00 88 0d       	or     $0xd880000,%eax
  403c88:	00 00                	add    %al,(%eax)
  403c8a:	84 0d 00 00 80 0d    	test   %cl,0xd800000
  403c90:	00 00                	add    %al,(%eax)
  403c92:	7c 0d                	jl     0x403ca1
  403c94:	00 00                	add    %al,(%eax)
  403c96:	a8 10                	test   $0x10,%al
  403c98:	00 00                	add    %al,(%eax)
  403c9a:	c5 10                	lds    (%eax),%edx
  403c9c:	00 00                	add    %al,(%eax)
  403c9e:	e2 10                	loop   0x403cb0
  403ca0:	00 00                	add    %al,(%eax)
  403ca2:	ff 10                	call   *(%eax)
  403ca4:	00 00                	add    %al,(%eax)
  403ca6:	58                   	pop    %eax
  403ca7:	0d 00 00 54 0d       	or     $0xd540000,%eax
  403cac:	00 00                	add    %al,(%eax)
  403cae:	be 0f 00 00 5c       	mov    $0x5c00000f,%esi
  403cb3:	0d 00 00 58 0d       	or     $0xd580000,%eax
  403cb8:	00 00                	add    %al,(%eax)
  403cba:	54                   	push   %esp
  403cbb:	0d 00 00 50 0d       	or     $0xd500000,%eax
  403cc0:	00 00                	add    %al,(%eax)
  403cc2:	4c                   	dec    %esp
  403cc3:	0d 00 00 a6 0f       	or     $0xfa60000,%eax
  403cc8:	00 00                	add    %al,(%eax)
  403cca:	44                   	inc    %esp
  403ccb:	0d 00 00 9e 0f       	or     $0xf9e0000,%eax
  403cd0:	00 00                	add    %al,(%eax)
  403cd2:	2c 0d                	sub    $0xd,%al
  403cd4:	00 00                	add    %al,(%eax)
  403cd6:	38 0d 00 00 92 0f    	cmp    %cl,0xf920000
  403cdc:	00 00                	add    %al,(%eax)
  403cde:	20 0d 00 00 1c 0d    	and    %cl,0xd1c0000
  403ce4:	00 00                	add    %al,(%eax)
  403ce6:	18 0d 00 00 14 0d    	sbb    %cl,0xd140000
  403cec:	00 00                	add    %al,(%eax)
  403cee:	10 0d 00 00 0c 0d    	adc    %cl,0xd0c0000
  403cf4:	00 00                	add    %al,(%eax)
  403cf6:	08 0d 00 00 04 0d    	or     %cl,0xd040000
  403cfc:	00 00                	add    %al,(%eax)
  403cfe:	85 14 00             	test   %edx,(%eax,%eax,1)
  403d01:	00 fc                	add    %bh,%ah
  403d03:	0c 00                	or     $0x0,%al
  403d05:	00 f8                	add    %bh,%al
  403d07:	0c 00                	or     $0x0,%al
  403d09:	00 f4                	add    %dh,%ah
  403d0b:	0c 00                	or     $0x0,%al
  403d0d:	00 f0                	add    %dh,%al
  403d0f:	0c 00                	or     $0x0,%al
  403d11:	00 ec                	add    %ch,%ah
  403d13:	0c 00                	or     $0x0,%al
  403d15:	00 e8                	add    %ch,%al
  403d17:	0c 00                	or     $0x0,%al
  403d19:	00 e4                	add    %ah,%ah
  403d1b:	0c 00                	or     $0x0,%al
  403d1d:	00 28                	add    %ch,(%eax)
  403d1f:	0e                   	push   %cs
  403d20:	00 00                	add    %al,(%eax)
  403d22:	24 0e                	and    $0xe,%al
  403d24:	00 00                	add    %al,(%eax)
  403d26:	20 0e                	and    %cl,(%esi)
  403d28:	00 00                	add    %al,(%eax)
  403d2a:	1c 0e                	sbb    $0xe,%al
  403d2c:	00 00                	add    %al,(%eax)
  403d2e:	3e 0f 00 00          	sldt   %ds:(%eax)
  403d32:	3a 0f                	cmp    (%edi),%cl
  403d34:	00 00                	add    %al,(%eax)
  403d36:	36 0f 00 00          	sldt   %ss:(%eax)
  403d3a:	32 0f                	xor    (%edi),%cl
  403d3c:	00 00                	add    %al,(%eax)
  403d3e:	c0 0c 00 00          	rorb   $0x0,(%eax,%eax,1)
  403d42:	bc 0c 00 00 b8       	mov    $0xb800000c,%esp
  403d47:	0c 00                	or     $0x0,%al
  403d49:	00 b4 0c 00 00 b0 0c 	add    %dh,0xcb00000(%esp,%ecx,1)
  403d50:	00 00                	add    %al,(%eax)
  403d52:	ac                   	lods   %ds:(%esi),%al
  403d53:	0c 00                	or     $0x0,%al
  403d55:	00 a8 0c 00 00 a4    	add    %ch,-0x5bfffff4(%eax)
  403d5b:	0c 00                	or     $0x0,%al
  403d5d:	00 b0 0c 00 00 ac    	add    %dh,-0x53fffff4(%eax)
  403d63:	0c 00                	or     $0x0,%al
  403d65:	00 a8 0c 00 00 a4    	add    %ch,-0x5bfffff4(%eax)
  403d6b:	0c 00                	or     $0x0,%al
  403d6d:	00 a0 0c 00 00 9c    	add    %ah,-0x63fffff4(%eax)
  403d73:	0c 00                	or     $0x0,%al
  403d75:	00 f6                	add    %dh,%dh
  403d77:	0e                   	push   %cs
  403d78:	00 00                	add    %al,(%eax)
  403d7a:	f2 0e                	repnz push %cs
  403d7c:	00 00                	add    %al,(%eax)
  403d7e:	bc 15 00 00 ea       	mov    $0xea000015,%esp
  403d83:	0e                   	push   %cs
  403d84:	00 00                	add    %al,(%eax)
  403d86:	db 15 00 00 e2 0e    	fistl  0xee20000
  403d8c:	00 00                	add    %al,(%eax)
  403d8e:	de 0e                	fimuls (%esi)
  403d90:	00 00                	add    %al,(%eax)
  403d92:	da 0e                	fimull (%esi)
  403d94:	00 00                	add    %al,(%eax)
  403d96:	d6                   	(bad)
  403d97:	0e                   	push   %cs
  403d98:	00 00                	add    %al,(%eax)
  403d9a:	d2 0e                	rorb   %cl,(%esi)
  403d9c:	00 00                	add    %al,(%eax)
  403d9e:	60                   	pusha
  403d9f:	0c 00                	or     $0x0,%al
  403da1:	00 5c 0c 00          	add    %bl,0x0(%esp,%ecx,1)
  403da5:	00 58 0c             	add    %bl,0xc(%eax)
  403da8:	00 00                	add    %al,(%eax)
  403daa:	54                   	push   %esp
  403dab:	0c 00                	or     $0x0,%al
  403dad:	00 50 0c             	add    %dl,0xc(%eax)
  403db0:	00 00                	add    %al,(%eax)
  403db2:	4c                   	dec    %esp
  403db3:	0c 00                	or     $0x0,%al
  403db5:	00 48 0c             	add    %cl,0xc(%eax)
  403db8:	00 00                	add    %al,(%eax)
  403dba:	44                   	inc    %esp
  403dbb:	0c 00                	or     $0x0,%al
  403dbd:	00 40 0c             	add    %al,0xc(%eax)
  403dc0:	00 00                	add    %al,(%eax)
  403dc2:	3c 0c                	cmp    $0xc,%al
  403dc4:	00 00                	add    %al,(%eax)
  403dc6:	38 0c 00             	cmp    %cl,(%eax,%eax,1)
  403dc9:	00 34 0c             	add    %dh,(%esp,%ecx,1)
  403dcc:	00 00                	add    %al,(%eax)
  403dce:	30 0c 00             	xor    %cl,(%eax,%eax,1)
  403dd1:	00 2c 0c             	add    %ch,(%esp,%ecx,1)
  403dd4:	00 00                	add    %al,(%eax)
  403dd6:	28 0c 00             	sub    %cl,(%eax,%eax,1)
  403dd9:	00 24 0c             	add    %ah,(%esp,%ecx,1)
  403ddc:	00 00                	add    %al,(%eax)
  403dde:	20 0c 00             	and    %cl,(%eax,%eax,1)
  403de1:	00 1c 0c             	add    %bl,(%esp,%ecx,1)
  403de4:	00 00                	add    %al,(%eax)
  403de6:	18 0c 00             	sbb    %cl,(%eax,%eax,1)
  403de9:	00 14 0c             	add    %dl,(%esp,%ecx,1)
  403dec:	00 00                	add    %al,(%eax)
  403dee:	10 0c 00             	adc    %cl,(%eax,%eax,1)
  403df1:	00 0c 0c             	add    %cl,(%esp,%ecx,1)
  403df4:	00 00                	add    %al,(%eax)
  403df6:	08 0c 00             	or     %cl,(%eax,%eax,1)
  403df9:	00 04 0c             	add    %al,(%esp,%ecx,1)
  403dfc:	00 00                	add    %al,(%eax)
  403dfe:	00 0c 00             	add    %cl,(%eax,%eax,1)
  403e01:	00 fc                	add    %bh,%ah
  403e03:	0b 00                	or     (%eax),%eax
  403e05:	00 f8                	add    %bh,%al
  403e07:	0b 00                	or     (%eax),%eax
  403e09:	00 f4                	add    %dh,%ah
  403e0b:	0b 00                	or     (%eax),%eax
  403e0d:	00 f0                	add    %dh,%al
  403e0f:	0b 00                	or     (%eax),%eax
  403e11:	00 ec                	add    %ch,%ah
  403e13:	0b 00                	or     (%eax),%eax
  403e15:	00 e8                	add    %ch,%al
  403e17:	0b 00                	or     (%eax),%eax
  403e19:	00 e4                	add    %ah,%ah
  403e1b:	0b 00                	or     (%eax),%eax
  403e1d:	00 e0                	add    %ah,%al
  403e1f:	0b 00                	or     (%eax),%eax
  403e21:	00 dc                	add    %bl,%ah
  403e23:	0b 00                	or     (%eax),%eax
  403e25:	00 d8                	add    %bl,%al
  403e27:	0b 00                	or     (%eax),%eax
  403e29:	00 d4                	add    %dl,%ah
  403e2b:	0b 00                	or     (%eax),%eax
  403e2d:	00 d0                	add    %dl,%al
  403e2f:	0b 00                	or     (%eax),%eax
  403e31:	00 cc                	add    %cl,%ah
  403e33:	0b 00                	or     (%eax),%eax
  403e35:	00 c8                	add    %cl,%al
  403e37:	0b 00                	or     (%eax),%eax
  403e39:	00 c4                	add    %al,%ah
  403e3b:	0b 00                	or     (%eax),%eax
  403e3d:	00 c0                	add    %al,%al
  403e3f:	0b 00                	or     (%eax),%eax
  403e41:	00 bc 0b 00 00 b8 0b 	add    %bh,0xbb80000(%ebx,%ecx,1)
  403e48:	00 00                	add    %al,(%eax)
  403e4a:	b4 0b                	mov    $0xb,%ah
  403e4c:	00 00                	add    %al,(%eax)
  403e4e:	c9                   	leave
  403e4f:	0b 00                	or     (%eax),%eax
  403e51:	00 c5                	add    %al,%ch
  403e53:	0b 00                	or     (%eax),%eax
  403e55:	00 a8 0b 00 00 a4    	add    %ch,-0x5bfffff5(%eax)
  403e5b:	0b 00                	or     (%eax),%eax
  403e5d:	00 a0 0b 00 00 8e    	add    %ah,-0x71fffff5(%eax)
  403e63:	10 00                	adc    %al,(%eax)
  403e65:	00 15 11 00 00 9c    	add    %dl,0x9c000011
  403e6b:	11 00                	adc    %eax,(%eax)
  403e6d:	00 90 0b 00 00 8c    	add    %dl,-0x73fffff5(%eax)
  403e73:	0b 00                	or     (%eax),%eax
  403e75:	00 88 0b 00 00 94    	add    %cl,-0x6bfffff5(%eax)
  403e7b:	0b 00                	or     (%eax),%eax
  403e7d:	00 80 0b 00 00 7c    	add    %al,0x7c00000b(%eax)
  403e83:	0b 00                	or     (%eax),%eax
  403e85:	00 e6                	add    %ah,%dh
  403e87:	0d 00 00 e2 0d       	or     $0xde20000,%eax
  403e8c:	00 00                	add    %al,(%eax)
  403e8e:	70 0b                	jo     0x403e9b
  403e90:	00 00                	add    %al,(%eax)
  403e92:	6c                   	insb   (%dx),%es:(%edi)
  403e93:	0b 00                	or     (%eax),%eax
  403e95:	00 68 0b             	add    %ch,0xb(%eax)
  403e98:	00 00                	add    %al,(%eax)
  403e9a:	64 0b 00             	or     %fs:(%eax),%eax
  403e9d:	00 b0 0b 00 00 ac    	add    %dh,-0x53fffff5(%eax)
  403ea3:	0b 00                	or     (%eax),%eax
  403ea5:	00 89 0c 00 00 85    	add    %cl,-0x7afffff4(%ecx)
  403eab:	0c 00                	or     $0x0,%al
  403ead:	00 81 0c 00 00 7d    	add    %al,0x7d00000c(%ecx)
  403eb3:	0c 00                	or     $0x0,%al
  403eb5:	00 98 0b 00 00 94    	add    %bl,-0x6bfffff5(%eax)
  403ebb:	0b 00                	or     (%eax),%eax
  403ebd:	00 90 0b 00 00 8c    	add    %dl,-0x73fffff5(%eax)
  403ec3:	0b 00                	or     (%eax),%eax
  403ec5:	00 88 0b 00 00 84    	add    %cl,-0x7bfffff5(%eax)
  403ecb:	0b 00                	or     (%eax),%eax
  403ecd:	00 80 0b 00 00 7c    	add    %al,0x7c00000b(%eax)
  403ed3:	0b 00                	or     (%eax),%eax
  403ed5:	00 78 0b             	add    %bh,0xb(%eax)
  403ed8:	00 00                	add    %al,(%eax)
  403eda:	74 0b                	je     0x403ee7
  403edc:	00 00                	add    %al,(%eax)
  403ede:	20 0b                	and    %cl,(%ebx)
  403ee0:	00 00                	add    %al,(%eax)
  403ee2:	1c 0b                	sbb    $0xb,%al
  403ee4:	00 00                	add    %al,(%eax)
  403ee6:	18 0b                	sbb    %cl,(%ebx)
  403ee8:	00 00                	add    %al,(%eax)
  403eea:	14 0b                	adc    $0xb,%al
  403eec:	00 00                	add    %al,(%eax)
  403eee:	10 0b                	adc    %cl,(%ebx)
  403ef0:	00 00                	add    %al,(%eax)
  403ef2:	0c 0b                	or     $0xb,%al
  403ef4:	00 00                	add    %al,(%eax)
  403ef6:	08 0b                	or     %cl,(%ebx)
  403ef8:	00 00                	add    %al,(%eax)
  403efa:	04 0b                	add    $0xb,%al
  403efc:	00 00                	add    %al,(%eax)
  403efe:	00 0b                	add    %cl,(%ebx)
  403f00:	00 00                	add    %al,(%eax)
  403f02:	fc                   	cld
  403f03:	0a 00                	or     (%eax),%al
  403f05:	00 f8                	add    %bh,%al
  403f07:	0a 00                	or     (%eax),%al
  403f09:	00 f4                	add    %dh,%ah
  403f0b:	0a 00                	or     (%eax),%al
  403f0d:	00 f0                	add    %dh,%al
  403f0f:	0a 00                	or     (%eax),%al
  403f11:	00 ec                	add    %ch,%ah
  403f13:	0a 00                	or     (%eax),%al
  403f15:	00 e8                	add    %ch,%al
  403f17:	0a 00                	or     (%eax),%al
  403f19:	00 e4                	add    %ah,%ah
  403f1b:	0a 00                	or     (%eax),%al
  403f1d:	00 f0                	add    %dh,%al
  403f1f:	0a 00                	or     (%eax),%al
  403f21:	00 ec                	add    %ch,%ah
  403f23:	0a 00                	or     (%eax),%al
  403f25:	00 e8                	add    %ch,%al
  403f27:	0a 00                	or     (%eax),%al
  403f29:	00 d4                	add    %dl,%ah
  403f2b:	0a 00                	or     (%eax),%al
  403f2d:	00 4b 0b             	add    %cl,0xb(%ebx)
  403f30:	00 00                	add    %al,(%eax)
  403f32:	cc                   	int3
  403f33:	0a 00                	or     (%eax),%al
  403f35:	00 36                	add    %dh,(%esi)
  403f37:	0d 00 00 32 0d       	or     $0xd320000,%eax
  403f3c:	00 00                	add    %al,(%eax)
  403f3e:	d0 0a                	rorb   $1,(%edx)
  403f40:	00 00                	add    %al,(%eax)
  403f42:	cc                   	int3
  403f43:	0a 00                	or     (%eax),%al
  403f45:	00 c8                	add    %cl,%al
  403f47:	0a 00                	or     (%eax),%al
  403f49:	00 b4 0a 00 00 2b 0b 	add    %dh,0xb2b0000(%edx,%ecx,1)
  403f50:	00 00                	add    %al,(%eax)
  403f52:	ac                   	lods   %ds:(%esi),%al
  403f53:	0a 00                	or     (%eax),%al
  403f55:	00 68 11             	add    %ch,0x11(%eax)
  403f58:	00 00                	add    %al,(%eax)
  403f5a:	a4                   	movsb  %ds:(%esi),%es:(%edi)
  403f5b:	0a 00                	or     (%eax),%al
  403f5d:	00 a0 0a 00 00 9c    	add    %ah,-0x63fffff6(%eax)
  403f63:	0a 00                	or     (%eax),%al
  403f65:	00 98 0a 00 00 94    	add    %bl,-0x6bfffff6(%eax)
  403f6b:	0a 00                	or     (%eax),%al
  403f6d:	00 90 0a 00 00 8c    	add    %dl,-0x73fffff6(%eax)
  403f73:	0a 00                	or     (%eax),%al
  403f75:	00 88 0a 00 00 84    	add    %cl,-0x7bfffff6(%eax)
  403f7b:	0a 00                	or     (%eax),%al
  403f7d:	00 80 0a 00 00 8c    	add    %al,-0x73fffff6(%eax)
  403f83:	0a 00                	or     (%eax),%al
  403f85:	00 ec                	add    %ch,%ah
  403f87:	0e                   	push   %cs
  403f88:	00 00                	add    %al,(%eax)
  403f8a:	74 0a                	je     0x403f96
  403f8c:	00 00                	add    %al,(%eax)
  403f8e:	70 0a                	jo     0x403f9a
  403f90:	00 00                	add    %al,(%eax)
  403f92:	6c                   	insb   (%dx),%es:(%edi)
  403f93:	0a 00                	or     (%eax),%al
  403f95:	00 68 0a             	add    %ch,0xa(%eax)
  403f98:	00 00                	add    %al,(%eax)
  403f9a:	64 0a 00             	or     %fs:(%eax),%al
  403f9d:	00 60 0a             	add    %ah,0xa(%eax)
  403fa0:	00 00                	add    %al,(%eax)
  403fa2:	5c                   	pop    %esp
  403fa3:	0a 00                	or     (%eax),%al
  403fa5:	00 58 0a             	add    %bl,0xa(%eax)
  403fa8:	00 00                	add    %al,(%eax)
  403faa:	54                   	push   %esp
  403fab:	0a 00                	or     (%eax),%al
  403fad:	00 50 0a             	add    %dl,0xa(%eax)
  403fb0:	00 00                	add    %al,(%eax)
  403fb2:	4c                   	dec    %esp
  403fb3:	0a 00                	or     (%eax),%al
  403fb5:	00 48 0a             	add    %cl,0xa(%eax)
  403fb8:	00 00                	add    %al,(%eax)
  403fba:	d9 0e                	(bad)  (%esi)
  403fbc:	00 00                	add    %al,(%eax)
  403fbe:	50                   	push   %eax
  403fbf:	0a 00                	or     (%eax),%al
  403fc1:	00 4c 0a 00          	add    %cl,0x0(%edx,%ecx,1)
  403fc5:	00 48 0a             	add    %cl,0xa(%eax)
  403fc8:	00 00                	add    %al,(%eax)
  403fca:	44                   	inc    %esp
  403fcb:	0a 00                	or     (%eax),%al
  403fcd:	00 40 0a             	add    %al,0xa(%eax)
  403fd0:	00 00                	add    %al,(%eax)
  403fd2:	3c 0a                	cmp    $0xa,%al
  403fd4:	00 00                	add    %al,(%eax)
  403fd6:	38 0a                	cmp    %cl,(%edx)
  403fd8:	00 00                	add    %al,(%eax)
  403fda:	34 0a                	xor    $0xa,%al
  403fdc:	00 00                	add    %al,(%eax)
  403fde:	20 0a                	and    %cl,(%edx)
  403fe0:	00 00                	add    %al,(%eax)
  403fe2:	1c 0a                	sbb    $0xa,%al
  403fe4:	00 00                	add    %al,(%eax)
  403fe6:	18 0a                	sbb    %cl,(%edx)
  403fe8:	00 00                	add    %al,(%eax)
  403fea:	14 0a                	adc    $0xa,%al
  403fec:	00 00                	add    %al,(%eax)
  403fee:	10 0a                	adc    %cl,(%edx)
  403ff0:	00 00                	add    %al,(%eax)
  403ff2:	0c 0a                	or     $0xa,%al
  403ff4:	00 00                	add    %al,(%eax)
  403ff6:	a9 0b 00 00 04       	test   $0x400000b,%eax
  403ffb:	0a 00                	or     (%eax),%al
  403ffd:	00 00                	add    %al,(%eax)
  403fff:	0a 00                	or     (%eax),%al
  404001:	00 fc                	add    %bh,%ah
  404003:	09 00                	or     %eax,(%eax)
  404005:	00 f8                	add    %bh,%al
  404007:	09 00                	or     %eax,(%eax)
  404009:	00 f4                	add    %dh,%ah
  40400b:	09 00                	or     %eax,(%eax)
  40400d:	00 f0                	add    %dh,%al
  40400f:	09 00                	or     %eax,(%eax)
  404011:	00 ec                	add    %ch,%ah
  404013:	09 00                	or     %eax,(%eax)
  404015:	00 e8                	add    %ch,%al
  404017:	09 00                	or     %eax,(%eax)
  404019:	00 e4                	add    %ah,%ah
  40401b:	09 00                	or     %eax,(%eax)
  40401d:	00 e0                	add    %ah,%al
  40401f:	09 00                	or     %eax,(%eax)
  404021:	00 dc                	add    %bl,%ah
  404023:	09 00                	or     %eax,(%eax)
  404025:	00 d8                	add    %bl,%al
  404027:	09 00                	or     %eax,(%eax)
  404029:	00 d4                	add    %dl,%ah
  40402b:	09 00                	or     %eax,(%eax)
  40402d:	00 d0                	add    %dl,%al
  40402f:	09 00                	or     %eax,(%eax)
  404031:	00 cc                	add    %cl,%ah
  404033:	09 00                	or     %eax,(%eax)
  404035:	00 69 0b             	add    %ch,0xb(%ecx)
  404038:	00 00                	add    %al,(%eax)
  40403a:	c4 09                	les    (%ecx),%ecx
  40403c:	00 00                	add    %al,(%eax)
  40403e:	c0 09 00             	rorb   $0x0,(%ecx)
  404041:	00 bc 09 00 00 b8 09 	add    %bh,0x9b80000(%ecx,%ecx,1)
  404048:	00 00                	add    %al,(%eax)
  40404a:	b4 09                	mov    $0x9,%ah
  40404c:	00 00                	add    %al,(%eax)
  40404e:	b0 09                	mov    $0x9,%al
  404050:	00 00                	add    %al,(%eax)
  404052:	ac                   	lods   %ds:(%esi),%al
  404053:	09 00                	or     %eax,(%eax)
  404055:	00 a8 09 00 00 a4    	add    %ch,-0x5bfffff7(%eax)
  40405b:	09 00                	or     %eax,(%eax)
  40405d:	00 fc                	add    %bh,%ah
  40405f:	0b 00                	or     (%eax),%eax
  404061:	00 9c 09 00 00 98 09 	add    %bl,0x9980000(%ecx,%ecx,1)
  404068:	00 00                	add    %al,(%eax)
  40406a:	94                   	xchg   %eax,%esp
  40406b:	09 00                	or     %eax,(%eax)
  40406d:	00 90 09 00 00 8c    	add    %dl,-0x73fffff7(%eax)
  404073:	09 00                	or     %eax,(%eax)
  404075:	00 88 09 00 00 84    	add    %cl,-0x7bfffff7(%eax)
  40407b:	09 00                	or     %eax,(%eax)
  40407d:	00 80 09 00 00 7c    	add    %al,0x7c000009(%eax)
  404083:	09 00                	or     %eax,(%eax)
  404085:	00 78 09             	add    %bh,0x9(%eax)
  404088:	00 00                	add    %al,(%eax)
  40408a:	74 09                	je     0x404095
  40408c:	00 00                	add    %al,(%eax)
  40408e:	70 09                	jo     0x404099
  404090:	00 00                	add    %al,(%eax)
  404092:	6c                   	insb   (%dx),%es:(%edi)
  404093:	09 00                	or     %eax,(%eax)
  404095:	00 68 09             	add    %ch,0x9(%eax)
  404098:	00 00                	add    %al,(%eax)
  40409a:	d2 0b                	rorb   %cl,(%ebx)
  40409c:	00 00                	add    %al,(%eax)
  40409e:	60                   	pusha
  40409f:	09 00                	or     %eax,(%eax)
  4040a1:	00 5c 09 00          	add    %bl,0x0(%ecx,%ecx,1)
  4040a5:	00 58 09             	add    %bl,0x9(%eax)
  4040a8:	00 00                	add    %al,(%eax)
  4040aa:	54                   	push   %esp
  4040ab:	09 00                	or     %eax,(%eax)
  4040ad:	00 50 09             	add    %dl,0x9(%eax)
  4040b0:	00 00                	add    %al,(%eax)
  4040b2:	4c                   	dec    %esp
  4040b3:	09 00                	or     %eax,(%eax)
  4040b5:	00 48 09             	add    %cl,0x9(%eax)
  4040b8:	00 00                	add    %al,(%eax)
  4040ba:	44                   	inc    %esp
  4040bb:	09 00                	or     %eax,(%eax)
  4040bd:	00 40 09             	add    %al,0x9(%eax)
  4040c0:	00 00                	add    %al,(%eax)
  4040c2:	3c 09                	cmp    $0x9,%al
  4040c4:	00 00                	add    %al,(%eax)
  4040c6:	38 09                	cmp    %cl,(%ecx)
  4040c8:	00 00                	add    %al,(%eax)
  4040ca:	34 09                	xor    $0x9,%al
  4040cc:	00 00                	add    %al,(%eax)
  4040ce:	9e                   	sahf
  4040cf:	0b 00                	or     (%eax),%eax
  4040d1:	00 9a 0b 00 00 96    	add    %bl,-0x69fffff5(%edx)
  4040d7:	0b 00                	or     (%eax),%eax
  4040d9:	00 92 0b 00 00 39    	add    %dl,0x3900000b(%edx)
  4040df:	09 00                	or     %eax,(%eax)
  4040e1:	00 8a 0b 00 00 86    	add    %cl,-0x79fffff5(%edx)
  4040e7:	0b 00                	or     (%eax),%eax
  4040e9:	00 82 0b 00 00 29    	add    %al,0x2900000b(%edx)
  4040ef:	09 00                	or     %eax,(%eax)
  4040f1:	00 25 09 00 00 76    	add    %ah,0x76000009
  4040f7:	0b 00                	or     (%eax),%eax
  4040f9:	00 1d 09 00 00 6e    	add    %bl,0x6e000009
  4040ff:	0b 00                	or     (%eax),%eax
  404101:	00 6a 0b             	add    %ch,0xb(%edx)
  404104:	00 00                	add    %al,(%eax)
  404106:	66 0b 00             	or     (%eax),%ax
  404109:	00 62 0b             	add    %ah,0xb(%edx)
  40410c:	00 00                	add    %al,(%eax)
  40410e:	f0 08 00             	lock or %al,(%eax)
  404111:	00 ec                	add    %ch,%ah
  404113:	08 00                	or     %al,(%eax)
  404115:	00 e8                	add    %ch,%al
  404117:	08 00                	or     %al,(%eax)
  404119:	00 52 0b             	add    %dl,0xb(%edx)
  40411c:	00 00                	add    %al,(%eax)
  40411e:	f9                   	stc
  40411f:	08 00                	or     %al,(%eax)
  404121:	00 f5                	add    %dh,%ch
  404123:	08 00                	or     %al,(%eax)
  404125:	00 f1                	add    %dh,%cl
  404127:	08 00                	or     %al,(%eax)
  404129:	00 ed                	add    %ch,%ch
  40412b:	08 00                	or     %al,(%eax)
  40412d:	00 e9                	add    %ch,%cl
  40412f:	08 00                	or     %al,(%eax)
  404131:	00 e5                	add    %ah,%ch
  404133:	08 00                	or     %al,(%eax)
  404135:	00 36                	add    %dh,(%esi)
  404137:	0b 00                	or     (%eax),%eax
  404139:	00 32                	add    %dh,(%edx)
  40413b:	0b 00                	or     (%eax),%eax
  40413d:	00 d9                	add    %bl,%cl
  40413f:	08 00                	or     %al,(%eax)
  404141:	00 d5                	add    %dl,%ch
  404143:	08 00                	or     %al,(%eax)
  404145:	00 d1                	add    %dl,%cl
  404147:	08 00                	or     %al,(%eax)
  404149:	00 cd                	add    %cl,%ch
  40414b:	08 00                	or     %al,(%eax)
  40414d:	00 1e                	add    %bl,(%esi)
  40414f:	0b 00                	or     (%eax),%eax
  404151:	00 1a                	add    %bl,(%edx)
  404153:	0b 00                	or     (%eax),%eax
  404155:	00 16                	add    %dl,(%esi)
  404157:	0b 00                	or     (%eax),%eax
  404159:	00 12                	add    %dl,(%edx)
  40415b:	0b 00                	or     (%eax),%eax
  40415d:	00 b9 08 00 00 b5    	add    %bh,-0x4afffff8(%ecx)
  404163:	08 00                	or     %al,(%eax)
  404165:	00 b1 08 00 00 ad    	add    %dh,-0x52fffff8(%ecx)
  40416b:	08 00                	or     %al,(%eax)
  40416d:	00 a9 08 00 00 a5    	add    %ch,-0x5afffff8(%ecx)
  404173:	08 00                	or     %al,(%eax)
  404175:	00 f6                	add    %dh,%dh
  404177:	0a 00                	or     (%eax),%al
  404179:	00 9d 08 00 00 99    	add    %bl,-0x66fffff8(%ebp)
  40417f:	08 00                	or     %al,(%eax)
  404181:	00 95 08 00 00 91    	add    %dl,-0x6efffff8(%ebp)
  404187:	08 00                	or     %al,(%eax)
  404189:	00 8d 08 00 00 89    	add    %cl,-0x76fffff8(%ebp)
  40418f:	08 00                	or     %al,(%eax)
  404191:	00 85 08 00 00 81    	add    %al,-0x7efffff8(%ebp)
  404197:	08 00                	or     %al,(%eax)
  404199:	00 7d 08             	add    %bh,0x8(%ebp)
  40419c:	00 00                	add    %al,(%eax)
  40419e:	79 08                	jns    0x4041a8
  4041a0:	00 00                	add    %al,(%eax)
  4041a2:	75 08                	jne    0x4041ac
  4041a4:	00 00                	add    %al,(%eax)
  4041a6:	c6                   	(bad)
  4041a7:	0a 00                	or     (%eax),%al
  4041a9:	00 c2                	add    %al,%dl
  4041ab:	0a 00                	or     (%eax),%al
  4041ad:	00 be 0a 00 00 ba    	add    %bh,-0x45fffff6(%esi)
  4041b3:	0a 00                	or     (%eax),%al
  4041b5:	00 b6 0a 00 00 b2    	add    %dh,-0x4dfffff6(%esi)
  4041bb:	0a 00                	or     (%eax),%al
  4041bd:	00 ae 0a 00 00 aa    	add    %ch,-0x55fffff6(%esi)
  4041c3:	0a 00                	or     (%eax),%al
  4041c5:	00 a6 0a 00 00 a2    	add    %ah,-0x5dfffff6(%esi)
  4041cb:	0a 00                	or     (%eax),%al
  4041cd:	00 9e 0a 00 00 9a    	add    %bl,-0x65fffff6(%esi)
  4041d3:	0a 00                	or     (%eax),%al
  4041d5:	00 96 0a 00 00 92    	add    %dl,-0x6dfffff6(%esi)
  4041db:	0a 00                	or     (%eax),%al
  4041dd:	00 8e 0a 00 00 8a    	add    %cl,-0x75fffff6(%esi)
  4041e3:	0a 00                	or     (%eax),%al
  4041e5:	00 86 0a 00 00 82    	add    %al,-0x7dfffff6(%esi)
  4041eb:	0a 00                	or     (%eax),%al
  4041ed:	00 7e 0a             	add    %bh,0xa(%esi)
  4041f0:	00 00                	add    %al,(%eax)
  4041f2:	7a 0a                	jp     0x4041fe
  4041f4:	00 00                	add    %al,(%eax)
  4041f6:	76 0a                	jbe    0x404202
  4041f8:	00 00                	add    %al,(%eax)
  4041fa:	72 0a                	jb     0x404206
  4041fc:	00 00                	add    %al,(%eax)
  4041fe:	6e                   	outsb  %ds:(%esi),(%dx)
  4041ff:	0a 00                	or     (%eax),%al
  404201:	00 6a 0a             	add    %ch,0xa(%edx)
  404204:	00 00                	add    %al,(%eax)
  404206:	66 0a 00             	data16 or (%eax),%al
  404209:	00 62 0a             	add    %ah,0xa(%edx)
  40420c:	00 00                	add    %al,(%eax)
  40420e:	5e                   	pop    %esi
  40420f:	0a 00                	or     (%eax),%al
  404211:	00 5a 0a             	add    %bl,0xa(%edx)
  404214:	00 00                	add    %al,(%eax)
  404216:	56                   	push   %esi
  404217:	0a 00                	or     (%eax),%al
  404219:	00 52 0a             	add    %dl,0xa(%edx)
  40421c:	00 00                	add    %al,(%eax)
  40421e:	4e                   	dec    %esi
  40421f:	0a 00                	or     (%eax),%al
  404221:	00 4a 0a             	add    %cl,0xa(%edx)
  404224:	00 00                	add    %al,(%eax)
  404226:	46                   	inc    %esi
  404227:	0a 00                	or     (%eax),%al
  404229:	00 42 0a             	add    %al,0xa(%edx)
  40422c:	00 00                	add    %al,(%eax)
  40422e:	3e 0a 00             	or     %ds:(%eax),%al
  404231:	00 3a                	add    %bh,(%edx)
  404233:	0a 00                	or     (%eax),%al
  404235:	00 36                	add    %dh,(%esi)
  404237:	0a 00                	or     (%eax),%al
  404239:	00 32                	add    %dh,(%edx)
  40423b:	0a 00                	or     (%eax),%al
  40423d:	00 2e                	add    %ch,(%esi)
  40423f:	0a 00                	or     (%eax),%al
  404241:	00 2a                	add    %ch,(%edx)
  404243:	0a 00                	or     (%eax),%al
  404245:	00 26                	add    %ah,(%esi)
  404247:	0a 00                	or     (%eax),%al
  404249:	00 22                	add    %ah,(%edx)
  40424b:	0a 00                	or     (%eax),%al
  40424d:	00 1e                	add    %bl,(%esi)
  40424f:	0a 00                	or     (%eax),%al
  404251:	00 1a                	add    %bl,(%edx)
  404253:	0a 00                	or     (%eax),%al
  404255:	00 16                	add    %dl,(%esi)
  404257:	0a 00                	or     (%eax),%al
  404259:	00 12                	add    %dl,(%edx)
  40425b:	0a 00                	or     (%eax),%al
  40425d:	00 0e                	add    %cl,(%esi)
  40425f:	0a 00                	or     (%eax),%al
  404261:	00 0a                	add    %cl,(%edx)
  404263:	0a 00                	or     (%eax),%al
  404265:	00 06                	add    %al,(%esi)
  404267:	0a 00                	or     (%eax),%al
  404269:	00 02                	add    %al,(%edx)
  40426b:	0a 00                	or     (%eax),%al
  40426d:	00 fe                	add    %bh,%dh
  40426f:	09 00                	or     %eax,(%eax)
  404271:	00 fa                	add    %bh,%dl
  404273:	09 00                	or     %eax,(%eax)
  404275:	00 f6                	add    %dh,%dh
  404277:	09 00                	or     %eax,(%eax)
  404279:	00 f2                	add    %dh,%dl
  40427b:	09 00                	or     %eax,(%eax)
  40427d:	00 ee                	add    %ch,%dh
  40427f:	09 00                	or     %eax,(%eax)
  404281:	00 ea                	add    %ch,%dl
  404283:	09 00                	or     %eax,(%eax)
  404285:	00 e6                	add    %ah,%dh
  404287:	09 00                	or     %eax,(%eax)
  404289:	00 e2                	add    %ah,%dl
  40428b:	09 00                	or     %eax,(%eax)
  40428d:	00 de                	add    %bl,%dh
  40428f:	09 00                	or     %eax,(%eax)
  404291:	00 da                	add    %bl,%dl
  404293:	09 00                	or     %eax,(%eax)
  404295:	00 d6                	add    %dl,%dh
  404297:	09 00                	or     %eax,(%eax)
  404299:	00 d2                	add    %dl,%dl
  40429b:	09 00                	or     %eax,(%eax)
  40429d:	00 ce                	add    %cl,%dh
  40429f:	09 00                	or     %eax,(%eax)
  4042a1:	00 ca                	add    %cl,%dl
  4042a3:	09 00                	or     %eax,(%eax)
  4042a5:	00 c6                	add    %al,%dh
  4042a7:	09 00                	or     %eax,(%eax)
  4042a9:	00 c2                	add    %al,%dl
  4042ab:	09 00                	or     %eax,(%eax)
  4042ad:	00 be 09 00 00 ba    	add    %bh,-0x45fffff7(%esi)
  4042b3:	09 00                	or     %eax,(%eax)
  4042b5:	00 b6 09 00 00 b2    	add    %dh,-0x4dfffff7(%esi)
  4042bb:	09 00                	or     %eax,(%eax)
  4042bd:	00 ae 09 00 00 aa    	add    %ch,-0x55fffff7(%esi)
  4042c3:	09 00                	or     %eax,(%eax)
  4042c5:	00 a6 09 00 00 a2    	add    %ah,-0x5dfffff7(%esi)
  4042cb:	09 00                	or     %eax,(%eax)
  4042cd:	00 9e 09 00 00 9a    	add    %bl,-0x65fffff7(%esi)
  4042d3:	09 00                	or     %eax,(%eax)
  4042d5:	00 96 09 00 00 92    	add    %dl,-0x6dfffff7(%esi)
  4042db:	09 00                	or     %eax,(%eax)
  4042dd:	00 8e 09 00 00 8a    	add    %cl,-0x75fffff7(%esi)
  4042e3:	09 00                	or     %eax,(%eax)
  4042e5:	00 86 09 00 00 82    	add    %al,-0x7dfffff7(%esi)
  4042eb:	09 00                	or     %eax,(%eax)
  4042ed:	00 7e 09             	add    %bh,0x9(%esi)
  4042f0:	00 00                	add    %al,(%eax)
  4042f2:	7a 09                	jp     0x4042fd
  4042f4:	00 00                	add    %al,(%eax)
  4042f6:	76 09                	jbe    0x404301
  4042f8:	00 00                	add    %al,(%eax)
  4042fa:	72 09                	jb     0x404305
  4042fc:	00 00                	add    %al,(%eax)
  4042fe:	6e                   	outsb  %ds:(%esi),(%dx)
  4042ff:	09 00                	or     %eax,(%eax)
  404301:	00 6a 09             	add    %ch,0x9(%edx)
  404304:	00 00                	add    %al,(%eax)
  404306:	66 09 00             	or     %ax,(%eax)
  404309:	00 62 09             	add    %ah,0x9(%edx)
  40430c:	00 00                	add    %al,(%eax)
  40430e:	5e                   	pop    %esi
  40430f:	09 00                	or     %eax,(%eax)
  404311:	00 5a 09             	add    %bl,0x9(%edx)
  404314:	00 00                	add    %al,(%eax)
  404316:	56                   	push   %esi
  404317:	09 00                	or     %eax,(%eax)
  404319:	00 52 09             	add    %dl,0x9(%edx)
  40431c:	00 00                	add    %al,(%eax)
  40431e:	4e                   	dec    %esi
  40431f:	09 00                	or     %eax,(%eax)
  404321:	00 4a 09             	add    %cl,0x9(%edx)
  404324:	00 00                	add    %al,(%eax)
  404326:	46                   	inc    %esi
  404327:	09 00                	or     %eax,(%eax)
  404329:	00 42 09             	add    %al,0x9(%edx)
  40432c:	00 00                	add    %al,(%eax)
  40432e:	3e 09 00             	or     %eax,%ds:(%eax)
  404331:	00 3a                	add    %bh,(%edx)
  404333:	09 00                	or     %eax,(%eax)
  404335:	00 36                	add    %dh,(%esi)
  404337:	09 00                	or     %eax,(%eax)
  404339:	00 32                	add    %dh,(%edx)
  40433b:	09 00                	or     %eax,(%eax)
  40433d:	00 2e                	add    %ch,(%esi)
  40433f:	09 00                	or     %eax,(%eax)
  404341:	00 2a                	add    %ch,(%edx)
  404343:	09 00                	or     %eax,(%eax)
  404345:	00 26                	add    %ah,(%esi)
  404347:	09 00                	or     %eax,(%eax)
  404349:	00 22                	add    %ah,(%edx)
  40434b:	09 00                	or     %eax,(%eax)
  40434d:	00 1e                	add    %bl,(%esi)
  40434f:	09 00                	or     %eax,(%eax)
  404351:	00 1a                	add    %bl,(%edx)
  404353:	09 00                	or     %eax,(%eax)
  404355:	00 16                	add    %dl,(%esi)
  404357:	09 00                	or     %eax,(%eax)
  404359:	00 12                	add    %dl,(%edx)
  40435b:	09 00                	or     %eax,(%eax)
  40435d:	00 0e                	add    %cl,(%esi)
  40435f:	09 00                	or     %eax,(%eax)
  404361:	00 0a                	add    %cl,(%edx)
  404363:	09 00                	or     %eax,(%eax)
  404365:	00 06                	add    %al,(%esi)
  404367:	09 00                	or     %eax,(%eax)
  404369:	00 02                	add    %al,(%edx)
  40436b:	09 00                	or     %eax,(%eax)
  40436d:	00 fe                	add    %bh,%dh
  40436f:	08 00                	or     %al,(%eax)
  404371:	00 fa                	add    %bh,%dl
  404373:	08 00                	or     %al,(%eax)
  404375:	00 f6                	add    %dh,%dh
  404377:	08 00                	or     %al,(%eax)
  404379:	00 f2                	add    %dh,%dl
  40437b:	08 00                	or     %al,(%eax)
  40437d:	00 ee                	add    %ch,%dh
  40437f:	08 00                	or     %al,(%eax)
  404381:	00 ea                	add    %ch,%dl
  404383:	08 00                	or     %al,(%eax)
  404385:	00 e6                	add    %ah,%dh
  404387:	08 00                	or     %al,(%eax)
  404389:	00 e2                	add    %ah,%dl
  40438b:	08 00                	or     %al,(%eax)
  40438d:	00 de                	add    %bl,%dh
  40438f:	08 00                	or     %al,(%eax)
  404391:	00 da                	add    %bl,%dl
  404393:	08 00                	or     %al,(%eax)
  404395:	00 d6                	add    %dl,%dh
  404397:	08 00                	or     %al,(%eax)
  404399:	00 d2                	add    %dl,%dl
  40439b:	08 00                	or     %al,(%eax)
  40439d:	00 ce                	add    %cl,%dh
  40439f:	08 00                	or     %al,(%eax)
  4043a1:	00 ca                	add    %cl,%dl
  4043a3:	08 00                	or     %al,(%eax)
  4043a5:	00 c6                	add    %al,%dh
  4043a7:	08 00                	or     %al,(%eax)
  4043a9:	00 c2                	add    %al,%dl
  4043ab:	08 00                	or     %al,(%eax)
  4043ad:	00 be 08 00 00 ba    	add    %bh,-0x45fffff8(%esi)
  4043b3:	08 00                	or     %al,(%eax)
  4043b5:	00 b6 08 00 00 b2    	add    %dh,-0x4dfffff8(%esi)
  4043bb:	08 00                	or     %al,(%eax)
  4043bd:	00 ae 08 00 00 aa    	add    %ch,-0x55fffff8(%esi)
  4043c3:	08 00                	or     %al,(%eax)
  4043c5:	00 a6 08 00 00 a2    	add    %ah,-0x5dfffff8(%esi)
  4043cb:	08 00                	or     %al,(%eax)
  4043cd:	00 9e 08 00 00 9a    	add    %bl,-0x65fffff8(%esi)
  4043d3:	08 00                	or     %al,(%eax)
  4043d5:	00 96 08 00 00 92    	add    %dl,-0x6dfffff8(%esi)
  4043db:	08 00                	or     %al,(%eax)
  4043dd:	00 8e 08 00 00 8a    	add    %cl,-0x75fffff8(%esi)
  4043e3:	08 00                	or     %al,(%eax)
  4043e5:	00 86 08 00 00 82    	add    %al,-0x7dfffff8(%esi)
  4043eb:	08 00                	or     %al,(%eax)
  4043ed:	00 7e 08             	add    %bh,0x8(%esi)
  4043f0:	00 00                	add    %al,(%eax)
  4043f2:	7a 08                	jp     0x4043fc
  4043f4:	00 00                	add    %al,(%eax)
  4043f6:	76 08                	jbe    0x404400
  4043f8:	00 00                	add    %al,(%eax)
  4043fa:	72 08                	jb     0x404404
  4043fc:	00 00                	add    %al,(%eax)
  4043fe:	6e                   	outsb  %ds:(%esi),(%dx)
  4043ff:	08 00                	or     %al,(%eax)
  404401:	00 6a 08             	add    %ch,0x8(%edx)
  404404:	00 00                	add    %al,(%eax)
  404406:	66 08 00             	data16 or %al,(%eax)
  404409:	00 62 08             	add    %ah,0x8(%edx)
  40440c:	00 00                	add    %al,(%eax)
  40440e:	5e                   	pop    %esi
  40440f:	08 00                	or     %al,(%eax)
  404411:	00 5a 08             	add    %bl,0x8(%edx)
  404414:	00 00                	add    %al,(%eax)
  404416:	56                   	push   %esi
  404417:	08 00                	or     %al,(%eax)
  404419:	00 52 08             	add    %dl,0x8(%edx)
  40441c:	00 00                	add    %al,(%eax)
  40441e:	4e                   	dec    %esi
  40441f:	08 00                	or     %al,(%eax)
  404421:	00 4a 08             	add    %cl,0x8(%edx)
  404424:	00 00                	add    %al,(%eax)
  404426:	46                   	inc    %esi
  404427:	08 00                	or     %al,(%eax)
  404429:	00 42 08             	add    %al,0x8(%edx)
  40442c:	00 00                	add    %al,(%eax)
  40442e:	3e 08 00             	or     %al,%ds:(%eax)
  404431:	00 3a                	add    %bh,(%edx)
  404433:	08 00                	or     %al,(%eax)
  404435:	00 36                	add    %dh,(%esi)
  404437:	08 00                	or     %al,(%eax)
  404439:	00 32                	add    %dh,(%edx)
  40443b:	08 00                	or     %al,(%eax)
  40443d:	00 2e                	add    %ch,(%esi)
  40443f:	08 00                	or     %al,(%eax)
  404441:	00 2a                	add    %ch,(%edx)
  404443:	08 00                	or     %al,(%eax)
  404445:	00 26                	add    %ah,(%esi)
  404447:	08 00                	or     %al,(%eax)
  404449:	00 22                	add    %ah,(%edx)
  40444b:	08 00                	or     %al,(%eax)
  40444d:	00 1e                	add    %bl,(%esi)
  40444f:	08 00                	or     %al,(%eax)
  404451:	00 1a                	add    %bl,(%edx)
  404453:	08 00                	or     %al,(%eax)
  404455:	00 16                	add    %dl,(%esi)
  404457:	08 00                	or     %al,(%eax)
  404459:	00 12                	add    %dl,(%edx)
  40445b:	08 00                	or     %al,(%eax)
  40445d:	00 2f                	add    %ch,(%edi)
  40445f:	07                   	pop    %es
  404460:	00 00                	add    %al,(%eax)
  404462:	2b 07                	sub    (%edi),%eax
  404464:	00 00                	add    %al,(%eax)
  404466:	06                   	push   %es
  404467:	08 00                	or     %al,(%eax)
  404469:	00 02                	add    %al,(%edx)
  40446b:	08 00                	or     %al,(%eax)
  40446d:	00 fe                	add    %bh,%dh
  40446f:	07                   	pop    %es
  404470:	00 00                	add    %al,(%eax)
  404472:	fa                   	cli
  404473:	07                   	pop    %es
  404474:	00 00                	add    %al,(%eax)
  404476:	f6 07 00             	testb  $0x0,(%edi)
  404479:	00 f2                	add    %dh,%dl
  40447b:	07                   	pop    %es
  40447c:	00 00                	add    %al,(%eax)
  40447e:	ee                   	out    %al,(%dx)
  40447f:	07                   	pop    %es
  404480:	00 00                	add    %al,(%eax)
  404482:	ea 07 00 00 e6 07 00 	ljmp   $0x7,$0xe6000007
  404489:	00 e2                	add    %ah,%dl
  40448b:	07                   	pop    %es
  40448c:	00 00                	add    %al,(%eax)
  40448e:	de 07                	fiadds (%edi)
  404490:	00 00                	add    %al,(%eax)
  404492:	da 07                	fiaddl (%edi)
  404494:	00 00                	add    %al,(%eax)
  404496:	d6                   	(bad)
  404497:	07                   	pop    %es
  404498:	00 00                	add    %al,(%eax)
  40449a:	d2 07                	rolb   %cl,(%edi)
  40449c:	00 00                	add    %al,(%eax)
  40449e:	ce                   	into
  40449f:	07                   	pop    %es
  4044a0:	00 00                	add    %al,(%eax)
  4044a2:	ca 07 00             	lret   $0x7
  4044a5:	00 c6                	add    %al,%dh
  4044a7:	07                   	pop    %es
  4044a8:	00 00                	add    %al,(%eax)
  4044aa:	c2 07 00             	ret    $0x7
  4044ad:	00 be 07 00 00 ba    	add    %bh,-0x45fffff9(%esi)
  4044b3:	07                   	pop    %es
  4044b4:	00 00                	add    %al,(%eax)
  4044b6:	b6 07                	mov    $0x7,%dh
  4044b8:	00 00                	add    %al,(%eax)
  4044ba:	b2 07                	mov    $0x7,%dl
  4044bc:	00 00                	add    %al,(%eax)
  4044be:	6b 05 00 00 67 05 00 	imul   $0x0,0x5670000,%eax
  4044c5:	00 63 05             	add    %ah,0x5(%ebx)
  4044c8:	00 00                	add    %al,(%eax)
  4044ca:	5f                   	pop    %edi
  4044cb:	05 00 00 5b 05       	add    $0x55b0000,%eax
  4044d0:	00 00                	add    %al,(%eax)
  4044d2:	57                   	push   %edi
  4044d3:	05 00 00 53 05       	add    $0x5530000,%eax
  4044d8:	00 00                	add    %al,(%eax)
  4044da:	24 05                	and    $0x5,%al
  4044dc:	00 00                	add    %al,(%eax)
  4044de:	8e 07                	mov    (%edi),%es
  4044e0:	00 00                	add    %al,(%eax)
  4044e2:	8a 07                	mov    (%edi),%al
  4044e4:	00 00                	add    %al,(%eax)
  4044e6:	86 07                	xchg   %al,(%edi)
  4044e8:	00 00                	add    %al,(%eax)
  4044ea:	82 07 00             	addb   $0x0,(%edi)
  4044ed:	00 3b                	add    %bh,(%ebx)
  4044ef:	05 00 00 37 05       	add    $0x5370000,%eax
  4044f4:	00 00                	add    %al,(%eax)
  4044f6:	33 05 00 00 2f 05    	xor    0x52f0000,%eax
  4044fc:	00 00                	add    %al,(%eax)
  4044fe:	6e                   	outsb  %ds:(%esi),(%dx)
  4044ff:	07                   	pop    %es
  404500:	00 00                	add    %al,(%eax)
  404502:	6a 07                	push   $0x7
  404504:	00 00                	add    %al,(%eax)
  404506:	66 07                	popw   %es
  404508:	00 00                	add    %al,(%eax)
  40450a:	62 07                	bound  %eax,(%edi)
  40450c:	00 00                	add    %al,(%eax)
  40450e:	5e                   	pop    %esi
  40450f:	07                   	pop    %es
  404510:	00 00                	add    %al,(%eax)
  404512:	5a                   	pop    %edx
  404513:	07                   	pop    %es
  404514:	00 00                	add    %al,(%eax)
  404516:	56                   	push   %esi
  404517:	07                   	pop    %es
  404518:	00 00                	add    %al,(%eax)
  40451a:	52                   	push   %edx
  40451b:	07                   	pop    %es
  40451c:	00 00                	add    %al,(%eax)
  40451e:	0b 05 00 00 07 05    	or     0x5070000,%eax
  404524:	00 00                	add    %al,(%eax)
  404526:	03 05 00 00 42 07    	add    0x7420000,%eax
  40452c:	00 00                	add    %al,(%eax)
  40452e:	3e 07                	ds pop %es
  404530:	00 00                	add    %al,(%eax)
  404532:	3a 07                	cmp    (%edi),%al
  404534:	00 00                	add    %al,(%eax)
  404536:	36 07                	ss pop %es
  404538:	00 00                	add    %al,(%eax)
  40453a:	32 07                	xor    (%edi),%al
  40453c:	00 00                	add    %al,(%eax)
  40453e:	2e 07                	cs pop %es
  404540:	00 00                	add    %al,(%eax)
  404542:	2a 07                	sub    (%edi),%al
  404544:	00 00                	add    %al,(%eax)
  404546:	26 07                	es pop %es
  404548:	00 00                	add    %al,(%eax)
  40454a:	22 07                	and    (%edi),%al
  40454c:	00 00                	add    %al,(%eax)
  40454e:	1e                   	push   %ds
  40454f:	07                   	pop    %es
  404550:	00 00                	add    %al,(%eax)
  404552:	1a 07                	sbb    (%edi),%al
  404554:	00 00                	add    %al,(%eax)
  404556:	16                   	push   %ss
  404557:	07                   	pop    %es
  404558:	00 00                	add    %al,(%eax)
  40455a:	12 07                	adc    (%edi),%al
  40455c:	00 00                	add    %al,(%eax)
  40455e:	0e                   	push   %cs
  40455f:	07                   	pop    %es
  404560:	00 00                	add    %al,(%eax)
  404562:	0a 07                	or     (%edi),%al
  404564:	00 00                	add    %al,(%eax)
  404566:	06                   	push   %es
  404567:	07                   	pop    %es
  404568:	00 00                	add    %al,(%eax)
  40456a:	02 07                	add    (%edi),%al
  40456c:	00 00                	add    %al,(%eax)
  40456e:	fe 06                	incb   (%esi)
  404570:	00 00                	add    %al,(%eax)
  404572:	fa                   	cli
  404573:	06                   	push   %es
  404574:	00 00                	add    %al,(%eax)
  404576:	f6 06 00             	testb  $0x0,(%esi)
  404579:	00 f2                	add    %dh,%dl
  40457b:	06                   	push   %es
  40457c:	00 00                	add    %al,(%eax)
  40457e:	ee                   	out    %al,(%dx)
  40457f:	06                   	push   %es
  404580:	00 00                	add    %al,(%eax)
  404582:	ea 06 00 00 e6 06 00 	ljmp   $0x6,$0xe6000006
  404589:	00 e2                	add    %ah,%dl
  40458b:	06                   	push   %es
  40458c:	00 00                	add    %al,(%eax)
  40458e:	de 06                	fiadds (%esi)
  404590:	00 00                	add    %al,(%eax)
  404592:	da 06                	fiaddl (%esi)
  404594:	00 00                	add    %al,(%eax)
  404596:	d6                   	(bad)
  404597:	06                   	push   %es
  404598:	00 00                	add    %al,(%eax)
  40459a:	d2 06                	rolb   %cl,(%esi)
  40459c:	00 00                	add    %al,(%eax)
  40459e:	8b 04 00             	mov    (%eax,%eax,1),%eax
  4045a1:	00 87 04 00 00 83    	add    %al,-0x7cfffffc(%edi)
  4045a7:	04 00                	add    $0x0,%al
  4045a9:	00 c2                	add    %al,%dl
  4045ab:	06                   	push   %es
  4045ac:	00 00                	add    %al,(%eax)
  4045ae:	be 06 00 00 ba       	mov    $0xba000006,%esi
  4045b3:	06                   	push   %es
  4045b4:	00 00                	add    %al,(%eax)
  4045b6:	b6 06                	mov    $0x6,%dh
  4045b8:	00 00                	add    %al,(%eax)
  4045ba:	b2 06                	mov    $0x6,%dl
  4045bc:	00 00                	add    %al,(%eax)
  4045be:	ae                   	scas   %es:(%edi),%al
  4045bf:	06                   	push   %es
  4045c0:	00 00                	add    %al,(%eax)
  4045c2:	aa                   	stos   %al,%es:(%edi)
  4045c3:	06                   	push   %es
  4045c4:	00 00                	add    %al,(%eax)
  4045c6:	a6                   	cmpsb  %es:(%edi),%ds:(%esi)
  4045c7:	06                   	push   %es
  4045c8:	00 00                	add    %al,(%eax)
  4045ca:	a2 06 00 00 9e       	mov    %al,0x9e000006
  4045cf:	06                   	push   %es
  4045d0:	00 00                	add    %al,(%eax)
  4045d2:	9a 06 00 00 96 06 00 	lcall  $0x6,$0x96000006
  4045d9:	00 92 06 00 00 8e    	add    %dl,-0x71fffffa(%edx)
  4045df:	06                   	push   %es
  4045e0:	00 00                	add    %al,(%eax)
  4045e2:	8a 06                	mov    (%esi),%al
  4045e4:	00 00                	add    %al,(%eax)
  4045e6:	86 06                	xchg   %al,(%esi)
  4045e8:	00 00                	add    %al,(%eax)
  4045ea:	82 06 00             	addb   $0x0,(%esi)
  4045ed:	00 7e 06             	add    %bh,0x6(%esi)
  4045f0:	00 00                	add    %al,(%eax)
  4045f2:	7a 06                	jp     0x4045fa
  4045f4:	00 00                	add    %al,(%eax)
  4045f6:	76 06                	jbe    0x4045fe
  4045f8:	00 00                	add    %al,(%eax)
  4045fa:	72 06                	jb     0x404602
  4045fc:	00 00                	add    %al,(%eax)
  4045fe:	6e                   	outsb  %ds:(%esi),(%dx)
  4045ff:	06                   	push   %es
  404600:	00 00                	add    %al,(%eax)
  404602:	6a 06                	push   $0x6
  404604:	00 00                	add    %al,(%eax)
  404606:	66 06                	pushw  %es
  404608:	00 00                	add    %al,(%eax)
  40460a:	62 06                	bound  %eax,(%esi)
  40460c:	00 00                	add    %al,(%eax)
  40460e:	5e                   	pop    %esi
  40460f:	06                   	push   %es
  404610:	00 00                	add    %al,(%eax)
  404612:	5a                   	pop    %edx
  404613:	06                   	push   %es
  404614:	00 00                	add    %al,(%eax)
  404616:	56                   	push   %esi
  404617:	06                   	push   %es
  404618:	00 00                	add    %al,(%eax)
  40461a:	52                   	push   %edx
  40461b:	06                   	push   %es
  40461c:	00 00                	add    %al,(%eax)
  40461e:	0b 04 00             	or     (%eax,%eax,1),%eax
  404621:	00 07                	add    %al,(%edi)
  404623:	04 00                	add    $0x0,%al
  404625:	00 03                	add    %al,(%ebx)
  404627:	04 00                	add    $0x0,%al
  404629:	00 ff                	add    %bh,%bh
  40462b:	03 00                	add    (%eax),%eax
  40462d:	00 3e                	add    %bh,(%esi)
  40462f:	06                   	push   %es
  404630:	00 00                	add    %al,(%eax)
  404632:	3a 06                	cmp    (%esi),%al
  404634:	00 00                	add    %al,(%eax)
  404636:	36 06                	ss push %es
  404638:	00 00                	add    %al,(%eax)
  40463a:	32 06                	xor    (%esi),%al
  40463c:	00 00                	add    %al,(%eax)
  40463e:	2e 06                	cs push %es
  404640:	00 00                	add    %al,(%eax)
  404642:	2a 06                	sub    (%esi),%al
  404644:	00 00                	add    %al,(%eax)
  404646:	26 06                	es push %es
  404648:	00 00                	add    %al,(%eax)
  40464a:	22 06                	and    (%esi),%al
  40464c:	00 00                	add    %al,(%eax)
  40464e:	1e                   	push   %ds
  40464f:	06                   	push   %es
  404650:	00 00                	add    %al,(%eax)
  404652:	1a 06                	sbb    (%esi),%al
  404654:	00 00                	add    %al,(%eax)
  404656:	16                   	push   %ss
  404657:	06                   	push   %es
  404658:	00 00                	add    %al,(%eax)
  40465a:	12 06                	adc    (%esi),%al
  40465c:	00 00                	add    %al,(%eax)
  40465e:	0e                   	push   %cs
  40465f:	06                   	push   %es
  404660:	00 00                	add    %al,(%eax)
  404662:	0a 06                	or     (%esi),%al
  404664:	00 00                	add    %al,(%eax)
  404666:	06                   	push   %es
  404667:	06                   	push   %es
  404668:	00 00                	add    %al,(%eax)
  40466a:	02 06                	add    (%esi),%al
  40466c:	00 00                	add    %al,(%eax)
  40466e:	fe 05 00 00 fa 05    	incb   0x5fa0000
  404674:	00 00                	add    %al,(%eax)
  404676:	f6 05 00 00 f2 05 00 	testb  $0x0,0x5f20000
  40467d:	00 ee                	add    %ch,%dh
  40467f:	05 00 00 ea 05       	add    $0x5ea0000,%eax
  404684:	00 00                	add    %al,(%eax)
  404686:	e6 05                	out    %al,$0x5
  404688:	00 00                	add    %al,(%eax)
  40468a:	e2 05                	loop   0x404691
  40468c:	00 00                	add    %al,(%eax)
  40468e:	de 05 00 00 da 05    	fiadds 0x5da0000
  404694:	00 00                	add    %al,(%eax)
  404696:	d6                   	(bad)
  404697:	05 00 00 d2 05       	add    $0x5d20000,%eax
  40469c:	00 00                	add    %al,(%eax)
  40469e:	ce                   	into
  40469f:	05 00 00 ca 05       	add    $0x5ca0000,%eax
  4046a4:	00 00                	add    %al,(%eax)
  4046a6:	c6 05 00 00 c2 05 00 	movb   $0x0,0x5c20000
  4046ad:	00 be 05 00 00 ba    	add    %bh,-0x45fffffb(%esi)
  4046b3:	05 00 00 b6 05       	add    $0x5b60000,%eax
  4046b8:	00 00                	add    %al,(%eax)
  4046ba:	b2 05                	mov    $0x5,%dl
  4046bc:	00 00                	add    %al,(%eax)
  4046be:	ae                   	scas   %es:(%edi),%al
  4046bf:	05 00 00 aa 05       	add    $0x5aa0000,%eax
  4046c4:	00 00                	add    %al,(%eax)
  4046c6:	a6                   	cmpsb  %es:(%edi),%ds:(%esi)
  4046c7:	05 00 00 a2 05       	add    $0x5a20000,%eax
  4046cc:	00 00                	add    %al,(%eax)
  4046ce:	9e                   	sahf
  4046cf:	05 00 00 9a 05       	add    $0x59a0000,%eax
  4046d4:	00 00                	add    %al,(%eax)
  4046d6:	96                   	xchg   %eax,%esi
  4046d7:	05 00 00 92 05       	add    $0x5920000,%eax
  4046dc:	00 00                	add    %al,(%eax)
  4046de:	8e 05 00 00 8a 05    	mov    0x58a0000,%es
  4046e4:	00 00                	add    %al,(%eax)
  4046e6:	86 05 00 00 82 05    	xchg   %al,0x5820000
  4046ec:	00 00                	add    %al,(%eax)
  4046ee:	7e 05                	jle    0x4046f5
  4046f0:	00 00                	add    %al,(%eax)
  4046f2:	7a 05                	jp     0x4046f9
  4046f4:	00 00                	add    %al,(%eax)
  4046f6:	76 05                	jbe    0x4046fd
  4046f8:	00 00                	add    %al,(%eax)
  4046fa:	72 05                	jb     0x404701
  4046fc:	00 00                	add    %al,(%eax)
  4046fe:	6e                   	outsb  %ds:(%esi),(%dx)
  4046ff:	05 00 00 6a 05       	add    $0x56a0000,%eax
  404704:	00 00                	add    %al,(%eax)
  404706:	66 05 00 00          	add    $0x0,%ax
  40470a:	62 05 00 00 5e 05    	bound  %eax,0x55e0000
  404710:	00 00                	add    %al,(%eax)
  404712:	5a                   	pop    %edx
  404713:	05 00 00 56 05       	add    $0x5560000,%eax
  404718:	00 00                	add    %al,(%eax)
  40471a:	52                   	push   %edx
  40471b:	05 00 00 4e 05       	add    $0x54e0000,%eax
  404720:	00 00                	add    %al,(%eax)
  404722:	4a                   	dec    %edx
  404723:	05 00 00 46 05       	add    $0x5460000,%eax
  404728:	00 00                	add    %al,(%eax)
  40472a:	42                   	inc    %edx
  40472b:	05 00 00 3e 05       	add    $0x53e0000,%eax
  404730:	00 00                	add    %al,(%eax)
  404732:	3a 05 00 00 36 05    	cmp    0x5360000,%al
  404738:	00 00                	add    %al,(%eax)
  40473a:	32 05 00 00 2e 05    	xor    0x52e0000,%al
  404740:	00 00                	add    %al,(%eax)
  404742:	2a 05 00 00 26 05    	sub    0x5260000,%al
  404748:	00 00                	add    %al,(%eax)
  40474a:	22 05 00 00 1e 05    	and    0x51e0000,%al
  404750:	00 00                	add    %al,(%eax)
  404752:	1a 05 00 00 16 05    	sbb    0x5160000,%al
  404758:	00 00                	add    %al,(%eax)
  40475a:	12 05 00 00 0e 05    	adc    0x50e0000,%al
  404760:	00 00                	add    %al,(%eax)
  404762:	0a 05 00 00 06 05    	or     0x5060000,%al
  404768:	00 00                	add    %al,(%eax)
  40476a:	02 05 00 00 fe 04    	add    0x4fe0000,%al
  404770:	00 00                	add    %al,(%eax)
  404772:	fa                   	cli
  404773:	04 00                	add    $0x0,%al
  404775:	00 f6                	add    %dh,%dh
  404777:	04 00                	add    $0x0,%al
  404779:	00 f2                	add    %dh,%dl
  40477b:	04 00                	add    $0x0,%al
  40477d:	00 ee                	add    %ch,%dh
  40477f:	04 00                	add    $0x0,%al
  404781:	00 ea                	add    %ch,%dl
  404783:	04 00                	add    $0x0,%al
  404785:	00 e6                	add    %ah,%dh
  404787:	04 00                	add    $0x0,%al
  404789:	00 e2                	add    %ah,%dl
  40478b:	04 00                	add    $0x0,%al
  40478d:	00 de                	add    %bl,%dh
  40478f:	04 00                	add    $0x0,%al
  404791:	00 da                	add    %bl,%dl
  404793:	04 00                	add    $0x0,%al
  404795:	00 d6                	add    %dl,%dh
  404797:	04 00                	add    $0x0,%al
  404799:	00 d2                	add    %dl,%dl
  40479b:	04 00                	add    $0x0,%al
  40479d:	00 ce                	add    %cl,%dh
  40479f:	04 00                	add    $0x0,%al
  4047a1:	00 ca                	add    %cl,%dl
  4047a3:	04 00                	add    $0x0,%al
  4047a5:	00 c6                	add    %al,%dh
  4047a7:	04 00                	add    $0x0,%al
  4047a9:	00 c2                	add    %al,%dl
  4047ab:	04 00                	add    $0x0,%al
  4047ad:	00 be 04 00 00 ba    	add    %bh,-0x45fffffc(%esi)
  4047b3:	04 00                	add    $0x0,%al
  4047b5:	00 b6 04 00 00 b2    	add    %dh,-0x4dfffffc(%esi)
  4047bb:	04 00                	add    $0x0,%al
  4047bd:	00 ae 04 00 00 aa    	add    %ch,-0x55fffffc(%esi)
  4047c3:	04 00                	add    $0x0,%al
  4047c5:	00 a6 04 00 00 a2    	add    %ah,-0x5dfffffc(%esi)
  4047cb:	04 00                	add    $0x0,%al
  4047cd:	00 9e 04 00 00 9a    	add    %bl,-0x65fffffc(%esi)
  4047d3:	04 00                	add    $0x0,%al
  4047d5:	00 96 04 00 00 92    	add    %dl,-0x6dfffffc(%esi)
  4047db:	04 00                	add    $0x0,%al
  4047dd:	00 8e 04 00 00 8a    	add    %cl,-0x75fffffc(%esi)
  4047e3:	04 00                	add    $0x0,%al
  4047e5:	00 86 04 00 00 82    	add    %al,-0x7dfffffc(%esi)
  4047eb:	04 00                	add    $0x0,%al
  4047ed:	00 7e 04             	add    %bh,0x4(%esi)
  4047f0:	00 00                	add    %al,(%eax)
  4047f2:	7a 04                	jp     0x4047f8
  4047f4:	00 00                	add    %al,(%eax)
  4047f6:	76 04                	jbe    0x4047fc
  4047f8:	00 00                	add    %al,(%eax)
  4047fa:	72 04                	jb     0x404800
  4047fc:	00 00                	add    %al,(%eax)
  4047fe:	6e                   	outsb  %ds:(%esi),(%dx)
  4047ff:	04 00                	add    $0x0,%al
  404801:	00 6a 04             	add    %ch,0x4(%edx)
  404804:	00 00                	add    %al,(%eax)
  404806:	66 04 00             	data16 add $0x0,%al
  404809:	00 62 04             	add    %ah,0x4(%edx)
  40480c:	00 00                	add    %al,(%eax)
  40480e:	5e                   	pop    %esi
  40480f:	04 00                	add    $0x0,%al
  404811:	00 5a 04             	add    %bl,0x4(%edx)
  404814:	00 00                	add    %al,(%eax)
  404816:	56                   	push   %esi
  404817:	04 00                	add    $0x0,%al
  404819:	00 52 04             	add    %dl,0x4(%edx)
  40481c:	00 00                	add    %al,(%eax)
  40481e:	4e                   	dec    %esi
  40481f:	04 00                	add    $0x0,%al
  404821:	00 4a 04             	add    %cl,0x4(%edx)
  404824:	00 00                	add    %al,(%eax)
  404826:	46                   	inc    %esi
  404827:	04 00                	add    $0x0,%al
  404829:	00 42 04             	add    %al,0x4(%edx)
  40482c:	00 00                	add    %al,(%eax)
  40482e:	3e 04 00             	ds add $0x0,%al
  404831:	00 3a                	add    %bh,(%edx)
  404833:	04 00                	add    $0x0,%al
  404835:	00 36                	add    %dh,(%esi)
  404837:	04 00                	add    $0x0,%al
  404839:	00 32                	add    %dh,(%edx)
  40483b:	04 00                	add    $0x0,%al
  40483d:	00 2e                	add    %ch,(%esi)
  40483f:	04 00                	add    $0x0,%al
  404841:	00 2a                	add    %ch,(%edx)
  404843:	04 00                	add    $0x0,%al
  404845:	00 26                	add    %ah,(%esi)
  404847:	04 00                	add    $0x0,%al
  404849:	00 22                	add    %ah,(%edx)
  40484b:	04 00                	add    $0x0,%al
  40484d:	00 1e                	add    %bl,(%esi)
  40484f:	04 00                	add    $0x0,%al
  404851:	00 1a                	add    %bl,(%edx)
  404853:	04 00                	add    $0x0,%al
  404855:	00 16                	add    %dl,(%esi)
  404857:	04 00                	add    $0x0,%al
  404859:	00 12                	add    %dl,(%edx)
  40485b:	04 00                	add    $0x0,%al
  40485d:	00 0e                	add    %cl,(%esi)
  40485f:	04 00                	add    $0x0,%al
  404861:	00 0a                	add    %cl,(%edx)
  404863:	04 00                	add    $0x0,%al
  404865:	00 06                	add    %al,(%esi)
  404867:	04 00                	add    $0x0,%al
  404869:	00 02                	add    %al,(%edx)
  40486b:	04 00                	add    $0x0,%al
  40486d:	00 fe                	add    %bh,%dh
  40486f:	03 00                	add    (%eax),%eax
  404871:	00 fa                	add    %bh,%dl
  404873:	03 00                	add    (%eax),%eax
  404875:	00 f6                	add    %dh,%dh
  404877:	03 00                	add    (%eax),%eax
  404879:	00 f2                	add    %dh,%dl
  40487b:	03 00                	add    (%eax),%eax
  40487d:	00 ee                	add    %ch,%dh
  40487f:	03 00                	add    (%eax),%eax
  404881:	00 ea                	add    %ch,%dl
  404883:	03 00                	add    (%eax),%eax
  404885:	00 e6                	add    %ah,%dh
  404887:	03 00                	add    (%eax),%eax
  404889:	00 e2                	add    %ah,%dl
  40488b:	03 00                	add    (%eax),%eax
  40488d:	00 de                	add    %bl,%dh
  40488f:	03 00                	add    (%eax),%eax
  404891:	00 da                	add    %bl,%dl
  404893:	03 00                	add    (%eax),%eax
  404895:	00 d6                	add    %dl,%dh
  404897:	03 00                	add    (%eax),%eax
  404899:	00 d2                	add    %dl,%dl
  40489b:	03 00                	add    (%eax),%eax
  40489d:	00 21                	add    %ah,(%ecx)
  40489f:	01 00                	add    %eax,(%eax)
  4048a1:	00 1d 01 00 00 19    	add    %bl,0x19000001
  4048a7:	01 00                	add    %eax,(%eax)
  4048a9:	00 15 01 00 00 12    	add    %dl,0x12000001
  4048af:	01 00                	add    %eax,(%eax)
  4048b1:	00 36                	add    %dh,(%esi)
  4048b3:	01 00                	add    %eax,(%eax)
  4048b5:	00 3d 01 00 00 05    	add    %bh,0x5000001
  4048bb:	01 00                	add    %eax,(%eax)
  4048bd:	00 01                	add    %al,(%ecx)
  4048bf:	01 00                	add    %eax,(%eax)
  4048c1:	00 fd                	add    %bh,%ch
  4048c3:	00 00                	add    %al,(%eax)
  4048c5:	00 f9                	add    %bh,%cl
  4048c7:	00 00                	add    %al,(%eax)
  4048c9:	00 f5                	add    %dh,%ch
  4048cb:	00 00                	add    %al,(%eax)
  4048cd:	00 f2                	add    %dh,%dl
  4048cf:	00 00                	add    %al,(%eax)
  4048d1:	00 ed                	add    %ch,%ch
  4048d3:	00 00                	add    %al,(%eax)
  4048d5:	00 e9                	add    %ch,%cl
  4048d7:	00 00                	add    %al,(%eax)
  4048d9:	00 e5                	add    %ah,%ch
  4048db:	00 00                	add    %al,(%eax)
  4048dd:	00 e1                	add    %ah,%cl
  4048df:	00 00                	add    %al,(%eax)
  4048e1:	00 dd                	add    %bl,%ch
  4048e3:	00 00                	add    %al,(%eax)
  4048e5:	00 d9                	add    %bl,%cl
  4048e7:	00 00                	add    %al,(%eax)
  4048e9:	00 d5                	add    %dl,%ch
  4048eb:	00 00                	add    %al,(%eax)
  4048ed:	00 d2                	add    %dl,%dl
  4048ef:	00 00                	add    %al,(%eax)
  4048f1:	00 cd                	add    %cl,%ch
  4048f3:	00 00                	add    %al,(%eax)
  4048f5:	00 c9                	add    %cl,%cl
  4048f7:	00 00                	add    %al,(%eax)
  4048f9:	00 c5                	add    %al,%ch
  4048fb:	00 00                	add    %al,(%eax)
  4048fd:	00 c1                	add    %al,%cl
  4048ff:	00 00                	add    %al,(%eax)
  404901:	00 bd 00 00 00 b9    	add    %bh,-0x47000000(%ebp)
  404907:	00 00                	add    %al,(%eax)
  404909:	00 b5 00 00 00 b1    	add    %dh,-0x4f000000(%ebp)
  40490f:	00 00                	add    %al,(%eax)
  404911:	00 ad 00 00 00 a9    	add    %ch,-0x57000000(%ebp)
  404917:	00 00                	add    %al,(%eax)
  404919:	00 a5 00 00 00 5e    	add    %ah,0x5e000000(%ebp)
  40491f:	ff 75 2f             	push   0x2f(%ebp)
  404922:	8f 45 23             	pop    0x23(%ebp)
  404925:	c6 45 22 00          	movb   $0x0,0x22(%ebp)
  404929:	c7 45 02 20 00 00 00 	movl   $0x20,0x2(%ebp)
  404930:	c7 45 06 20 00 00 00 	movl   $0x20,0x6(%ebp)
  404937:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  40493b:	75 07                	jne    0x404944
  40493d:	c7 45 06 40 00 00 00 	movl   $0x40,0x6(%ebp)
  404944:	8b 45 23             	mov    0x23(%ebp),%eax
  404947:	0f b6 08             	movzbl (%eax),%ecx
  40494a:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  40494d:	03 00                	add    (%eax),%eax
  40494f:	83 c0 04             	add    $0x4,%eax
  404952:	ff d0                	call   *%eax
  404954:	83 f8 ff             	cmp    $0xffffffff,%eax
  404957:	74 06                	je     0x40495f
  404959:	8b 45 23             	mov    0x23(%ebp),%eax
  40495c:	2b 45 2f             	sub    0x2f(%ebp),%eax
  40495f:	5e                   	pop    %esi
  404960:	5a                   	pop    %edx
  404961:	59                   	pop    %ecx
  404962:	83 c4 27             	add    $0x27,%esp
  404965:	5d                   	pop    %ebp
  404966:	c2 08 00             	ret    $0x8
  404969:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  404970:	8b 45 23             	mov    0x23(%ebp),%eax
  404973:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  404977:	25 c7 00 00 00       	and    $0xc7,%eax
  40497c:	b9 40 00 00 00       	mov    $0x40,%ecx
  404981:	31 d2                	xor    %edx,%edx
  404983:	f7 f1                	div    %ecx
  404985:	89 45 0a             	mov    %eax,0xa(%ebp)
  404988:	83 f8 01             	cmp    $0x1,%eax
  40498b:	75 04                	jne    0x404991
  40498d:	83 45 1a 01          	addl   $0x1,0x1a(%ebp)
  404991:	83 f8 02             	cmp    $0x2,%eax
  404994:	75 04                	jne    0x40499a
  404996:	83 45 1a 04          	addl   $0x4,0x1a(%ebp)
  40499a:	89 55 0e             	mov    %edx,0xe(%ebp)
  40499d:	c1 e0 05             	shl    $0x5,%eax
  4049a0:	01 f0                	add    %esi,%eax
  4049a2:	05 00 10 00 00       	add    $0x1000,%eax
  4049a7:	8d 04 90             	lea    (%eax,%edx,4),%eax
  4049aa:	03 00                	add    (%eax),%eax
  4049ac:	83 c0 04             	add    $0x4,%eax
  4049af:	ff d0                	call   *%eax
  4049b1:	c3                   	ret
  4049b2:	8b 45 23             	mov    0x23(%ebp),%eax
  4049b5:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  4049b9:	83 e0 38             	and    $0x38,%eax
  4049bc:	c1 e8 03             	shr    $0x3,%eax
  4049bf:	89 45 16             	mov    %eax,0x16(%ebp)
  4049c2:	c3                   	ret
  4049c3:	c3                   	ret
  4049c4:	83 7d 06 20          	cmpl   $0x20,0x6(%ebp)
  4049c8:	7c 21                	jl     0x4049eb
  4049ca:	83 45 1a 01          	addl   $0x1,0x1a(%ebp)
  4049ce:	8b 45 23             	mov    0x23(%ebp),%eax
  4049d1:	0f b6 40 02          	movzbl 0x2(%eax),%eax
  4049d5:	83 e0 07             	and    $0x7,%eax
  4049d8:	89 45 12             	mov    %eax,0x12(%ebp)
  4049db:	83 7d 12 05          	cmpl   $0x5,0x12(%ebp)
  4049df:	75 0a                	jne    0x4049eb
  4049e1:	83 7d 0a 00          	cmpl   $0x0,0xa(%ebp)
  4049e5:	75 04                	jne    0x4049eb
  4049e7:	83 45 1a 04          	addl   $0x4,0x1a(%ebp)
  4049eb:	c3                   	ret
  4049ec:	83 7d 06 20          	cmpl   $0x20,0x6(%ebp)
  4049f0:	7c 04                	jl     0x4049f6
  4049f2:	83 45 1a 04          	addl   $0x4,0x1a(%ebp)
  4049f6:	c3                   	ret
  4049f7:	83 7d 06 10          	cmpl   $0x10,0x6(%ebp)
  4049fb:	75 04                	jne    0x404a01
  4049fd:	83 45 1a 02          	addl   $0x2,0x1a(%ebp)
  404a01:	c3                   	ret
  404a02:	e8 62 ff ff ff       	call   0x404969
  404a07:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404a0a:	01 45 23             	add    %eax,0x23(%ebp)
  404a0d:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404a11:	c3                   	ret
  404a12:	ff 45 23             	incl   0x23(%ebp)
  404a15:	c3                   	ret
  404a16:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404a1a:	c3                   	ret
  404a1b:	83 7d 02 10          	cmpl   $0x10,0x2(%ebp)
  404a1f:	75 06                	jne    0x404a27
  404a21:	e8 dc ff ff ff       	call   0x404a02
  404a26:	c3                   	ret
  404a27:	e8 44 02 00 00       	call   0x404c70
  404a2c:	c3                   	ret
  404a2d:	83 7d 02 10          	cmpl   $0x10,0x2(%ebp)
  404a31:	75 09                	jne    0x404a3c
  404a33:	e8 ca ff ff ff       	call   0x404a02
  404a38:	ff 45 23             	incl   0x23(%ebp)
  404a3b:	c3                   	ret
  404a3c:	e8 2f 02 00 00       	call   0x404c70
  404a41:	c3                   	ret
  404a42:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404a46:	75 06                	jne    0x404a4e
  404a48:	e8 23 02 00 00       	call   0x404c70
  404a4d:	c3                   	ret
  404a4e:	ff 45 23             	incl   0x23(%ebp)
  404a51:	c3                   	ret
  404a52:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404a56:	7c 05                	jl     0x404a5d
  404a58:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404a5c:	c3                   	ret
  404a5d:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404a61:	c3                   	ret
  404a62:	83 7d 02 40          	cmpl   $0x40,0x2(%ebp)
  404a66:	75 05                	jne    0x404a6d
  404a68:	83 45 23 09          	addl   $0x9,0x23(%ebp)
  404a6c:	c3                   	ret
  404a6d:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404a71:	75 05                	jne    0x404a78
  404a73:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404a77:	c3                   	ret
  404a78:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404a7c:	c3                   	ret
  404a7d:	e8 80 ff ff ff       	call   0x404a02
  404a82:	ff 45 23             	incl   0x23(%ebp)
  404a85:	c3                   	ret
  404a86:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404a8a:	75 22                	jne    0x404aae
  404a8c:	c7 45 02 40 00 00 00 	movl   $0x40,0x2(%ebp)
  404a93:	ff 45 23             	incl   0x23(%ebp)
  404a96:	8b 45 23             	mov    0x23(%ebp),%eax
  404a99:	0f b6 08             	movzbl (%eax),%ecx
  404a9c:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  404a9f:	03 00                	add    (%eax),%eax
  404aa1:	83 c0 04             	add    $0x4,%eax
  404aa4:	ff d0                	call   *%eax
  404aa6:	c7 45 02 20 00 00 00 	movl   $0x20,0x2(%ebp)
  404aad:	c3                   	ret
  404aae:	ff 45 23             	incl   0x23(%ebp)
  404ab1:	c3                   	ret
  404ab2:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404ab6:	75 23                	jne    0x404adb
  404ab8:	ff 45 23             	incl   0x23(%ebp)
  404abb:	fe 45 22             	incb   0x22(%ebp)
  404abe:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  404ac2:	75 06                	jne    0x404aca
  404ac4:	e8 a7 01 00 00       	call   0x404c70
  404ac9:	c3                   	ret
  404aca:	8b 45 23             	mov    0x23(%ebp),%eax
  404acd:	0f b6 08             	movzbl (%eax),%ecx
  404ad0:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  404ad3:	03 00                	add    (%eax),%eax
  404ad5:	83 c0 04             	add    $0x4,%eax
  404ad8:	ff d0                	call   *%eax
  404ada:	c3                   	ret
  404adb:	83 45 23 01          	addl   $0x1,0x23(%ebp)
  404adf:	c3                   	ret
  404ae0:	ff 45 23             	incl   0x23(%ebp)
  404ae3:	fe 45 22             	incb   0x22(%ebp)
  404ae6:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  404aea:	75 06                	jne    0x404af2
  404aec:	e8 7f 01 00 00       	call   0x404c70
  404af1:	c3                   	ret
  404af2:	8b 45 23             	mov    0x23(%ebp),%eax
  404af5:	0f b6 08             	movzbl (%eax),%ecx
  404af8:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  404afb:	03 00                	add    (%eax),%eax
  404afd:	83 c0 04             	add    $0x4,%eax
  404b00:	ff d0                	call   *%eax
  404b02:	c3                   	ret
  404b03:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404b07:	7c 0a                	jl     0x404b13
  404b09:	e8 f4 fe ff ff       	call   0x404a02
  404b0e:	83 45 23 04          	addl   $0x4,0x23(%ebp)
  404b12:	c3                   	ret
  404b13:	e8 ea fe ff ff       	call   0x404a02
  404b18:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404b1c:	c3                   	ret
  404b1d:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404b21:	75 06                	jne    0x404b29
  404b23:	e8 48 01 00 00       	call   0x404c70
  404b28:	c3                   	ret
  404b29:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404b2d:	c3                   	ret
  404b2e:	83 45 23 04          	addl   $0x4,0x23(%ebp)
  404b32:	c3                   	ret
  404b33:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404b37:	c3                   	ret
  404b38:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404b3c:	75 06                	jne    0x404b44
  404b3e:	e8 2d 01 00 00       	call   0x404c70
  404b43:	c3                   	ret
  404b44:	e8 b9 fe ff ff       	call   0x404a02
  404b49:	c3                   	ret
  404b4a:	e8 1a fe ff ff       	call   0x404969
  404b4f:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404b53:	75 06                	jne    0x404b5b
  404b55:	e8 a8 fe ff ff       	call   0x404a02
  404b5a:	c3                   	ret
  404b5b:	e8 10 01 00 00       	call   0x404c70
  404b60:	c3                   	ret
  404b61:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404b65:	c3                   	ret
  404b66:	83 7d 06 40          	cmpl   $0x40,0x6(%ebp)
  404b6a:	75 05                	jne    0x404b71
  404b6c:	83 45 23 09          	addl   $0x9,0x23(%ebp)
  404b70:	c3                   	ret
  404b71:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404b75:	c3                   	ret
  404b76:	83 7d 06 10          	cmpl   $0x10,0x6(%ebp)
  404b7a:	75 05                	jne    0x404b81
  404b7c:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404b80:	c3                   	ret
  404b81:	83 7d 06 20          	cmpl   $0x20,0x6(%ebp)
  404b85:	75 05                	jne    0x404b8c
  404b87:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404b8b:	c3                   	ret
  404b8c:	83 45 23 09          	addl   $0x9,0x23(%ebp)
  404b90:	c3                   	ret
  404b91:	80 7d 00 01          	cmpb   $0x1,0x0(%ebp)
  404b95:	75 06                	jne    0x404b9d
  404b97:	e8 66 fe ff ff       	call   0x404a02
  404b9c:	c3                   	ret
  404b9d:	e8 ce 00 00 00       	call   0x404c70
  404ba2:	c3                   	ret
  404ba3:	80 7d 00 01          	cmpb   $0x1,0x0(%ebp)
  404ba7:	75 06                	jne    0x404baf
  404ba9:	e8 54 fe ff ff       	call   0x404a02
  404bae:	c3                   	ret
  404baf:	80 7d 01 01          	cmpb   $0x1,0x1(%ebp)
  404bb3:	75 06                	jne    0x404bbb
  404bb5:	e8 48 fe ff ff       	call   0x404a02
  404bba:	c3                   	ret
  404bbb:	83 7d 02 10          	cmpl   $0x10,0x2(%ebp)
  404bbf:	75 06                	jne    0x404bc7
  404bc1:	e8 3c fe ff ff       	call   0x404a02
  404bc6:	c3                   	ret
  404bc7:	e8 a4 00 00 00       	call   0x404c70
  404bcc:	c3                   	ret
  404bcd:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404bd1:	75 06                	jne    0x404bd9
  404bd3:	e8 98 00 00 00       	call   0x404c70
  404bd8:	c3                   	ret
  404bd9:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404bdd:	75 05                	jne    0x404be4
  404bdf:	83 45 23 07          	addl   $0x7,0x23(%ebp)
  404be3:	c3                   	ret
  404be4:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404be8:	c3                   	ret
  404be9:	c3                   	ret
  404bea:	83 7d 02 10          	cmpl   $0x10,0x2(%ebp)
  404bee:	74 10                	je     0x404c00
  404bf0:	e8 74 fd ff ff       	call   0x404969
  404bf5:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404bf8:	01 45 23             	add    %eax,0x23(%ebp)
  404bfb:	83 45 23 06          	addl   $0x6,0x23(%ebp)
  404bff:	c3                   	ret
  404c00:	e8 64 fd ff ff       	call   0x404969
  404c05:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404c08:	01 45 23             	add    %eax,0x23(%ebp)
  404c0b:	83 45 23 04          	addl   $0x4,0x23(%ebp)
  404c0f:	c3                   	ret
  404c10:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404c14:	75 06                	jne    0x404c1c
  404c16:	e8 55 00 00 00       	call   0x404c70
  404c1b:	c3                   	ret
  404c1c:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404c20:	75 05                	jne    0x404c27
  404c22:	83 45 23 07          	addl   $0x7,0x23(%ebp)
  404c26:	c3                   	ret
  404c27:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404c2b:	c3                   	ret
  404c2c:	e8 81 fd ff ff       	call   0x4049b2
  404c31:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404c35:	75 06                	jne    0x404c3d
  404c37:	e8 c6 fd ff ff       	call   0x404a02
  404c3c:	c3                   	ret
  404c3d:	e8 2e 00 00 00       	call   0x404c70
  404c42:	c3                   	ret
  404c43:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404c47:	75 05                	jne    0x404c4e
  404c49:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404c4d:	c3                   	ret
  404c4e:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404c52:	75 05                	jne    0x404c59
  404c54:	83 45 23 05          	addl   $0x5,0x23(%ebp)
  404c58:	c3                   	ret
  404c59:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404c5d:	c3                   	ret
  404c5e:	80 7d 00 01          	cmpb   $0x1,0x0(%ebp)
  404c62:	75 06                	jne    0x404c6a
  404c64:	e8 99 fd ff ff       	call   0x404a02
  404c69:	c3                   	ret
  404c6a:	e8 01 00 00 00       	call   0x404c70
  404c6f:	c3                   	ret
  404c70:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  404c75:	c3                   	ret
  404c76:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404c7a:	75 06                	jne    0x404c82
  404c7c:	e8 ef ff ff ff       	call   0x404c70
  404c81:	c3                   	ret
  404c82:	e8 7b fd ff ff       	call   0x404a02
  404c87:	83 45 23 01          	addl   $0x1,0x23(%ebp)
  404c8b:	c3                   	ret
  404c8c:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404c90:	7c 0a                	jl     0x404c9c
  404c92:	e8 6b fd ff ff       	call   0x404a02
  404c97:	83 45 23 04          	addl   $0x4,0x23(%ebp)
  404c9b:	c3                   	ret
  404c9c:	e8 61 fd ff ff       	call   0x404a02
  404ca1:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404ca5:	c3                   	ret
  404ca6:	e8 be fc ff ff       	call   0x404969
  404cab:	e8 02 fd ff ff       	call   0x4049b2
  404cb0:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404cb4:	75 0b                	jne    0x404cc1
  404cb6:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404cb9:	01 45 23             	add    %eax,0x23(%ebp)
  404cbc:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404cc0:	c3                   	ret
  404cc1:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  404cc5:	75 06                	jne    0x404ccd
  404cc7:	e8 a4 ff ff ff       	call   0x404c70
  404ccc:	c3                   	ret
  404ccd:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404cd0:	01 45 23             	add    %eax,0x23(%ebp)
  404cd3:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404cd7:	c3                   	ret
  404cd8:	83 7d 02 20          	cmpl   $0x20,0x2(%ebp)
  404cdc:	7c 32                	jl     0x404d10
  404cde:	e8 86 fc ff ff       	call   0x404969
  404ce3:	e8 ca fc ff ff       	call   0x4049b2
  404ce8:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404cec:	75 0b                	jne    0x404cf9
  404cee:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404cf1:	01 45 23             	add    %eax,0x23(%ebp)
  404cf4:	83 45 23 06          	addl   $0x6,0x23(%ebp)
  404cf8:	c3                   	ret
  404cf9:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  404cfd:	75 06                	jne    0x404d05
  404cff:	e8 6c ff ff ff       	call   0x404c70
  404d04:	c3                   	ret
  404d05:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404d08:	01 45 23             	add    %eax,0x23(%ebp)
  404d0b:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404d0f:	c3                   	ret
  404d10:	e8 54 fc ff ff       	call   0x404969
  404d15:	e8 98 fc ff ff       	call   0x4049b2
  404d1a:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404d1e:	75 0b                	jne    0x404d2b
  404d20:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404d23:	01 45 23             	add    %eax,0x23(%ebp)
  404d26:	83 45 23 04          	addl   $0x4,0x23(%ebp)
  404d2a:	c3                   	ret
  404d2b:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  404d2f:	75 06                	jne    0x404d37
  404d31:	e8 3a ff ff ff       	call   0x404c70
  404d36:	c3                   	ret
  404d37:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404d3a:	01 45 23             	add    %eax,0x23(%ebp)
  404d3d:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404d41:	c3                   	ret
  404d42:	e8 22 fc ff ff       	call   0x404969
  404d47:	e8 66 fc ff ff       	call   0x4049b2
  404d4c:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  404d50:	7e 06                	jle    0x404d58
  404d52:	e8 19 ff ff ff       	call   0x404c70
  404d57:	c3                   	ret
  404d58:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404d5b:	01 45 23             	add    %eax,0x23(%ebp)
  404d5e:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404d62:	c3                   	ret
  404d63:	e8 4a fc ff ff       	call   0x4049b2
  404d68:	83 7d 16 06          	cmpl   $0x6,0x16(%ebp)
  404d6c:	7e 06                	jle    0x404d74
  404d6e:	e8 fd fe ff ff       	call   0x404c70
  404d73:	c3                   	ret
  404d74:	e8 f0 fb ff ff       	call   0x404969
  404d79:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404d7c:	01 45 23             	add    %eax,0x23(%ebp)
  404d7f:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404d83:	c3                   	ret
  404d84:	e8 e0 fb ff ff       	call   0x404969
  404d89:	e8 24 fc ff ff       	call   0x4049b2
  404d8e:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  404d92:	7e 06                	jle    0x404d9a
  404d94:	e8 d7 fe ff ff       	call   0x404c70
  404d99:	c3                   	ret
  404d9a:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404d9d:	01 45 23             	add    %eax,0x23(%ebp)
  404da0:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404da4:	c3                   	ret
  404da5:	e8 bf fb ff ff       	call   0x404969
  404daa:	e8 03 fc ff ff       	call   0x4049b2
  404daf:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404db3:	75 1a                	jne    0x404dcf
  404db5:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404db9:	0f 85 ac 00 00 00    	jne    0x404e6b
  404dbf:	83 7d 0e 04          	cmpl   $0x4,0xe(%ebp)
  404dc3:	0f 8e a2 00 00 00    	jle    0x404e6b
  404dc9:	e8 a2 fe ff ff       	call   0x404c70
  404dce:	c3                   	ret
  404dcf:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  404dd3:	75 1a                	jne    0x404def
  404dd5:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404dd9:	0f 85 8c 00 00 00    	jne    0x404e6b
  404ddf:	83 7d 0e 01          	cmpl   $0x1,0xe(%ebp)
  404de3:	0f 8e 82 00 00 00    	jle    0x404e6b
  404de9:	e8 82 fe ff ff       	call   0x404c70
  404dee:	c3                   	ret
  404def:	83 7d 16 02          	cmpl   $0x2,0x16(%ebp)
  404df3:	75 10                	jne    0x404e05
  404df5:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404df9:	0f 85 6c 00 00 00    	jne    0x404e6b
  404dff:	e8 6c fe ff ff       	call   0x404c70
  404e04:	c3                   	ret
  404e05:	83 7d 16 03          	cmpl   $0x3,0x16(%ebp)
  404e09:	75 0c                	jne    0x404e17
  404e0b:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404e0f:	75 5a                	jne    0x404e6b
  404e11:	e8 5a fe ff ff       	call   0x404c70
  404e16:	c3                   	ret
  404e17:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  404e1b:	75 0c                	jne    0x404e29
  404e1d:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404e21:	75 48                	jne    0x404e6b
  404e23:	e8 48 fe ff ff       	call   0x404c70
  404e28:	c3                   	ret
  404e29:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  404e2d:	75 06                	jne    0x404e35
  404e2f:	e8 3c fe ff ff       	call   0x404c70
  404e34:	c3                   	ret
  404e35:	83 7d 16 06          	cmpl   $0x6,0x16(%ebp)
  404e39:	75 0c                	jne    0x404e47
  404e3b:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404e3f:	75 2a                	jne    0x404e6b
  404e41:	e8 2a fe ff ff       	call   0x404c70
  404e46:	c3                   	ret
  404e47:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  404e4b:	75 1e                	jne    0x404e6b
  404e4d:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404e51:	75 18                	jne    0x404e6b
  404e53:	83 7d 33 40          	cmpl   $0x40,0x33(%ebp)
  404e57:	75 0c                	jne    0x404e65
  404e59:	83 7d 0e 00          	cmpl   $0x0,0xe(%ebp)
  404e5d:	74 0c                	je     0x404e6b
  404e5f:	e8 0c fe ff ff       	call   0x404c70
  404e64:	c3                   	ret
  404e65:	e8 06 fe ff ff       	call   0x404c70
  404e6a:	c3                   	ret
  404e6b:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404e6e:	01 45 23             	add    %eax,0x23(%ebp)
  404e71:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404e75:	c3                   	ret
  404e76:	e8 ee fa ff ff       	call   0x404969
  404e7b:	e8 32 fb ff ff       	call   0x4049b2
  404e80:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  404e84:	7d 06                	jge    0x404e8c
  404e86:	e8 e5 fd ff ff       	call   0x404c70
  404e8b:	c3                   	ret
  404e8c:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404e8f:	01 45 23             	add    %eax,0x23(%ebp)
  404e92:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404e96:	c3                   	ret
  404e97:	e8 cd fa ff ff       	call   0x404969
  404e9c:	e8 11 fb ff ff       	call   0x4049b2
  404ea1:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404ea5:	75 06                	jne    0x404ead
  404ea7:	e8 c4 fd ff ff       	call   0x404c70
  404eac:	c3                   	ret
  404ead:	83 7d 16 02          	cmpl   $0x2,0x16(%ebp)
  404eb1:	75 06                	jne    0x404eb9
  404eb3:	e8 b8 fd ff ff       	call   0x404c70
  404eb8:	c3                   	ret
  404eb9:	83 7d 16 03          	cmpl   $0x3,0x16(%ebp)
  404ebd:	75 06                	jne    0x404ec5
  404ebf:	e8 ac fd ff ff       	call   0x404c70
  404ec4:	c3                   	ret
  404ec5:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  404ec9:	75 06                	jne    0x404ed1
  404ecb:	e8 a0 fd ff ff       	call   0x404c70
  404ed0:	c3                   	ret
  404ed1:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  404ed5:	75 06                	jne    0x404edd
  404ed7:	e8 94 fd ff ff       	call   0x404c70
  404edc:	c3                   	ret
  404edd:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  404ee1:	7e 06                	jle    0x404ee9
  404ee3:	e8 88 fd ff ff       	call   0x404c70
  404ee8:	c3                   	ret
  404ee9:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404eec:	01 45 23             	add    %eax,0x23(%ebp)
  404eef:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  404ef3:	c3                   	ret
  404ef4:	e8 b9 fa ff ff       	call   0x4049b2
  404ef9:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404efd:	75 06                	jne    0x404f05
  404eff:	e8 6c fd ff ff       	call   0x404c70
  404f04:	c3                   	ret
  404f05:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  404f09:	75 06                	jne    0x404f11
  404f0b:	e8 60 fd ff ff       	call   0x404c70
  404f10:	c3                   	ret
  404f11:	83 7d 16 02          	cmpl   $0x2,0x16(%ebp)
  404f15:	75 11                	jne    0x404f28
  404f17:	e8 4d fa ff ff       	call   0x404969
  404f1c:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404f20:	74 52                	je     0x404f74
  404f22:	e8 49 fd ff ff       	call   0x404c70
  404f27:	c3                   	ret
  404f28:	83 7d 16 03          	cmpl   $0x3,0x16(%ebp)
  404f2c:	75 06                	jne    0x404f34
  404f2e:	e8 3d fd ff ff       	call   0x404c70
  404f33:	c3                   	ret
  404f34:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  404f38:	75 11                	jne    0x404f4b
  404f3a:	e8 2a fa ff ff       	call   0x404969
  404f3f:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404f43:	74 2f                	je     0x404f74
  404f45:	e8 26 fd ff ff       	call   0x404c70
  404f4a:	c3                   	ret
  404f4b:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  404f4f:	75 06                	jne    0x404f57
  404f51:	e8 1a fd ff ff       	call   0x404c70
  404f56:	c3                   	ret
  404f57:	83 7d 16 06          	cmpl   $0x6,0x16(%ebp)
  404f5b:	75 11                	jne    0x404f6e
  404f5d:	e8 07 fa ff ff       	call   0x404969
  404f62:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404f66:	74 0c                	je     0x404f74
  404f68:	e8 03 fd ff ff       	call   0x404c70
  404f6d:	c3                   	ret
  404f6e:	e8 fd fc ff ff       	call   0x404c70
  404f73:	c3                   	ret
  404f74:	8b 45 1a             	mov    0x1a(%ebp),%eax
  404f77:	01 45 23             	add    %eax,0x23(%ebp)
  404f7a:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  404f7e:	c3                   	ret
  404f7f:	e8 2e fa ff ff       	call   0x4049b2
  404f84:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  404f88:	75 06                	jne    0x404f90
  404f8a:	e8 e1 fc ff ff       	call   0x404c70
  404f8f:	c3                   	ret
  404f90:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  404f94:	75 06                	jne    0x404f9c
  404f96:	e8 d5 fc ff ff       	call   0x404c70
  404f9b:	c3                   	ret
  404f9c:	83 7d 16 02          	cmpl   $0x2,0x16(%ebp)
  404fa0:	75 11                	jne    0x404fb3
  404fa2:	e8 c2 f9 ff ff       	call   0x404969
  404fa7:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404fab:	74 52                	je     0x404fff
  404fad:	e8 be fc ff ff       	call   0x404c70
  404fb2:	c3                   	ret
  404fb3:	83 7d 16 03          	cmpl   $0x3,0x16(%ebp)
  404fb7:	75 06                	jne    0x404fbf
  404fb9:	e8 b2 fc ff ff       	call   0x404c70
  404fbe:	c3                   	ret
  404fbf:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  404fc3:	75 11                	jne    0x404fd6
  404fc5:	e8 9f f9 ff ff       	call   0x404969
  404fca:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404fce:	74 2f                	je     0x404fff
  404fd0:	e8 9b fc ff ff       	call   0x404c70
  404fd5:	c3                   	ret
  404fd6:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  404fda:	75 06                	jne    0x404fe2
  404fdc:	e8 8f fc ff ff       	call   0x404c70
  404fe1:	c3                   	ret
  404fe2:	83 7d 16 06          	cmpl   $0x6,0x16(%ebp)
  404fe6:	75 11                	jne    0x404ff9
  404fe8:	e8 7c f9 ff ff       	call   0x404969
  404fed:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  404ff1:	74 0c                	je     0x404fff
  404ff3:	e8 78 fc ff ff       	call   0x404c70
  404ff8:	c3                   	ret
  404ff9:	e8 72 fc ff ff       	call   0x404c70
  404ffe:	c3                   	ret
  404fff:	8b 45 1a             	mov    0x1a(%ebp),%eax
  405002:	01 45 23             	add    %eax,0x23(%ebp)
  405005:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  405009:	c3                   	ret
  40500a:	e8 a3 f9 ff ff       	call   0x4049b2
  40500f:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  405013:	75 06                	jne    0x40501b
  405015:	e8 56 fc ff ff       	call   0x404c70
  40501a:	c3                   	ret
  40501b:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  40501f:	75 06                	jne    0x405027
  405021:	e8 4a fc ff ff       	call   0x404c70
  405026:	c3                   	ret
  405027:	83 7d 16 02          	cmpl   $0x2,0x16(%ebp)
  40502b:	75 15                	jne    0x405042
  40502d:	e8 37 f9 ff ff       	call   0x404969
  405032:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  405036:	0f 84 7b 00 00 00    	je     0x4050b7
  40503c:	e8 2f fc ff ff       	call   0x404c70
  405041:	c3                   	ret
  405042:	83 7d 16 03          	cmpl   $0x3,0x16(%ebp)
  405046:	75 1d                	jne    0x405065
  405048:	83 7d 02 10          	cmpl   $0x10,0x2(%ebp)
  40504c:	75 11                	jne    0x40505f
  40504e:	e8 16 f9 ff ff       	call   0x404969
  405053:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  405057:	74 5e                	je     0x4050b7
  405059:	e8 12 fc ff ff       	call   0x404c70
  40505e:	c3                   	ret
  40505f:	e8 0c fc ff ff       	call   0x404c70
  405064:	c3                   	ret
  405065:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  405069:	75 06                	jne    0x405071
  40506b:	e8 00 fc ff ff       	call   0x404c70
  405070:	c3                   	ret
  405071:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  405075:	75 06                	jne    0x40507d
  405077:	e8 f4 fb ff ff       	call   0x404c70
  40507c:	c3                   	ret
  40507d:	83 7d 16 06          	cmpl   $0x6,0x16(%ebp)
  405081:	75 11                	jne    0x405094
  405083:	e8 e1 f8 ff ff       	call   0x404969
  405088:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  40508c:	74 29                	je     0x4050b7
  40508e:	e8 dd fb ff ff       	call   0x404c70
  405093:	c3                   	ret
  405094:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  405098:	75 17                	jne    0x4050b1
  40509a:	83 7d 02 10          	cmpl   $0x10,0x2(%ebp)
  40509e:	75 11                	jne    0x4050b1
  4050a0:	e8 c4 f8 ff ff       	call   0x404969
  4050a5:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  4050a9:	74 0c                	je     0x4050b7
  4050ab:	e8 c0 fb ff ff       	call   0x404c70
  4050b0:	c3                   	ret
  4050b1:	e8 ba fb ff ff       	call   0x404c70
  4050b6:	c3                   	ret
  4050b7:	8b 45 1a             	mov    0x1a(%ebp),%eax
  4050ba:	01 45 23             	add    %eax,0x23(%ebp)
  4050bd:	83 45 23 03          	addl   $0x3,0x23(%ebp)
  4050c1:	c3                   	ret
  4050c2:	e8 eb f8 ff ff       	call   0x4049b2
  4050c7:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  4050cb:	75 15                	jne    0x4050e2
  4050cd:	e8 97 f8 ff ff       	call   0x404969
  4050d2:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  4050d6:	0f 85 a0 00 00 00    	jne    0x40517c
  4050dc:	e8 8f fb ff ff       	call   0x404c70
  4050e1:	c3                   	ret
  4050e2:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  4050e6:	75 15                	jne    0x4050fd
  4050e8:	e8 7c f8 ff ff       	call   0x404969
  4050ed:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  4050f1:	0f 85 85 00 00 00    	jne    0x40517c
  4050f7:	e8 74 fb ff ff       	call   0x404c70
  4050fc:	c3                   	ret
  4050fd:	83 7d 16 02          	cmpl   $0x2,0x16(%ebp)
  405101:	75 15                	jne    0x405118
  405103:	e8 61 f8 ff ff       	call   0x404969
  405108:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  40510c:	0f 85 6a 00 00 00    	jne    0x40517c
  405112:	e8 59 fb ff ff       	call   0x404c70
  405117:	c3                   	ret
  405118:	83 7d 16 03          	cmpl   $0x3,0x16(%ebp)
  40511c:	75 11                	jne    0x40512f
  40511e:	e8 46 f8 ff ff       	call   0x404969
  405123:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  405127:	75 53                	jne    0x40517c
  405129:	e8 42 fb ff ff       	call   0x404c70
  40512e:	c3                   	ret
  40512f:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  405133:	75 06                	jne    0x40513b
  405135:	e8 36 fb ff ff       	call   0x404c70
  40513a:	c3                   	ret
  40513b:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  40513f:	75 11                	jne    0x405152
  405141:	e8 23 f8 ff ff       	call   0x404969
  405146:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  40514a:	75 30                	jne    0x40517c
  40514c:	e8 1f fb ff ff       	call   0x404c70
  405151:	c3                   	ret
  405152:	83 7d 16 06          	cmpl   $0x6,0x16(%ebp)
  405156:	75 11                	jne    0x405169
  405158:	e8 0c f8 ff ff       	call   0x404969
  40515d:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  405161:	75 19                	jne    0x40517c
  405163:	e8 08 fb ff ff       	call   0x404c70
  405168:	c3                   	ret
  405169:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  40516d:	7f 07                	jg     0x405176
  40516f:	e8 f5 f7 ff ff       	call   0x404969
  405174:	eb 06                	jmp    0x40517c
  405176:	e8 f5 fa ff ff       	call   0x404c70
  40517b:	c3                   	ret
  40517c:	8b 45 1a             	mov    0x1a(%ebp),%eax
  40517f:	01 45 23             	add    %eax,0x23(%ebp)
  405182:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  405186:	c3                   	ret
  405187:	e8 26 f8 ff ff       	call   0x4049b2
  40518c:	83 7d 16 00          	cmpl   $0x0,0x16(%ebp)
  405190:	75 11                	jne    0x4051a3
  405192:	e8 d2 f7 ff ff       	call   0x404969
  405197:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  40519b:	75 51                	jne    0x4051ee
  40519d:	e8 ce fa ff ff       	call   0x404c70
  4051a2:	c3                   	ret
  4051a3:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  4051a7:	75 11                	jne    0x4051ba
  4051a9:	e8 bb f7 ff ff       	call   0x404969
  4051ae:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  4051b2:	75 3a                	jne    0x4051ee
  4051b4:	e8 b7 fa ff ff       	call   0x404c70
  4051b9:	c3                   	ret
  4051ba:	83 7d 16 02          	cmpl   $0x2,0x16(%ebp)
  4051be:	75 11                	jne    0x4051d1
  4051c0:	e8 a4 f7 ff ff       	call   0x404969
  4051c5:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  4051c9:	75 23                	jne    0x4051ee
  4051cb:	e8 a0 fa ff ff       	call   0x404c70
  4051d0:	c3                   	ret
  4051d1:	83 7d 16 03          	cmpl   $0x3,0x16(%ebp)
  4051d5:	75 11                	jne    0x4051e8
  4051d7:	e8 8d f7 ff ff       	call   0x404969
  4051dc:	83 7d 0a 03          	cmpl   $0x3,0xa(%ebp)
  4051e0:	75 0c                	jne    0x4051ee
  4051e2:	e8 89 fa ff ff       	call   0x404c70
  4051e7:	c3                   	ret
  4051e8:	e8 83 fa ff ff       	call   0x404c70
  4051ed:	c3                   	ret
  4051ee:	8b 45 1a             	mov    0x1a(%ebp),%eax
  4051f1:	01 45 23             	add    %eax,0x23(%ebp)
  4051f4:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  4051f8:	c3                   	ret
  4051f9:	ff 45 23             	incl   0x23(%ebp)
  4051fc:	c7 45 02 10 00 00 00 	movl   $0x10,0x2(%ebp)
  405203:	fe 45 22             	incb   0x22(%ebp)
  405206:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  40520a:	75 06                	jne    0x405212
  40520c:	e8 5f fa ff ff       	call   0x404c70
  405211:	c3                   	ret
  405212:	8b 45 23             	mov    0x23(%ebp),%eax
  405215:	0f b6 08             	movzbl (%eax),%ecx
  405218:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  40521b:	03 00                	add    (%eax),%eax
  40521d:	83 c0 04             	add    $0x4,%eax
  405220:	ff d0                	call   *%eax
  405222:	c7 45 02 20 00 00 00 	movl   $0x20,0x2(%ebp)
  405229:	c3                   	ret
  40522a:	ff 45 23             	incl   0x23(%ebp)
  40522d:	fe 45 22             	incb   0x22(%ebp)
  405230:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  405234:	75 06                	jne    0x40523c
  405236:	e8 35 fa ff ff       	call   0x404c70
  40523b:	c3                   	ret
  40523c:	8b 4d 06             	mov    0x6(%ebp),%ecx
  40523f:	d1 e9                	shr    $1,%ecx
  405241:	89 5d 06             	mov    %ebx,0x6(%ebp)
  405244:	8b 45 23             	mov    0x23(%ebp),%eax
  405247:	0f b6 08             	movzbl (%eax),%ecx
  40524a:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  40524d:	03 00                	add    (%eax),%eax
  40524f:	83 c0 04             	add    $0x4,%eax
  405252:	ff d0                	call   *%eax
  405254:	8b 5d 06             	mov    0x6(%ebp),%ebx
  405257:	d1 e1                	shl    $1,%ecx
  405259:	89 4d 06             	mov    %ecx,0x6(%ebp)
  40525c:	c3                   	ret
  40525d:	ff 45 23             	incl   0x23(%ebp)
  405260:	fe 45 22             	incb   0x22(%ebp)
  405263:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  405267:	75 06                	jne    0x40526f
  405269:	e8 02 fa ff ff       	call   0x404c70
  40526e:	c3                   	ret
  40526f:	8b 45 23             	mov    0x23(%ebp),%eax
  405272:	0f b6 00             	movzbl (%eax),%eax
  405275:	3c a4                	cmp    $0xa4,%al
  405277:	74 12                	je     0x40528b
  405279:	3c a7                	cmp    $0xa7,%al
  40527b:	74 0e                	je     0x40528b
  40527d:	3c ae                	cmp    $0xae,%al
  40527f:	74 0a                	je     0x40528b
  405281:	3c af                	cmp    $0xaf,%al
  405283:	74 06                	je     0x40528b
  405285:	3c 0f                	cmp    $0xf,%al
  405287:	74 02                	je     0x40528b
  405289:	eb 04                	jmp    0x40528f
  40528b:	c6 45 00 01          	movb   $0x1,0x0(%ebp)
  40528f:	8b 45 23             	mov    0x23(%ebp),%eax
  405292:	0f b6 08             	movzbl (%eax),%ecx
  405295:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  405298:	03 00                	add    (%eax),%eax
  40529a:	83 c0 04             	add    $0x4,%eax
  40529d:	ff d0                	call   *%eax
  40529f:	c6 45 00 00          	movb   $0x0,0x0(%ebp)
  4052a3:	c3                   	ret
  4052a4:	ff 45 23             	incl   0x23(%ebp)
  4052a7:	fe 45 22             	incb   0x22(%ebp)
  4052aa:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  4052ae:	75 06                	jne    0x4052b6
  4052b0:	e8 bb f9 ff ff       	call   0x404c70
  4052b5:	c3                   	ret
  4052b6:	8b 45 23             	mov    0x23(%ebp),%eax
  4052b9:	0f b6 00             	movzbl (%eax),%eax
  4052bc:	3c 90                	cmp    $0x90,%al
  4052be:	74 3e                	je     0x4052fe
  4052c0:	3c a4                	cmp    $0xa4,%al
  4052c2:	74 3a                	je     0x4052fe
  4052c4:	3c a5                	cmp    $0xa5,%al
  4052c6:	74 36                	je     0x4052fe
  4052c8:	3c a6                	cmp    $0xa6,%al
  4052ca:	74 32                	je     0x4052fe
  4052cc:	3c a7                	cmp    $0xa7,%al
  4052ce:	74 2e                	je     0x4052fe
  4052d0:	3c aa                	cmp    $0xaa,%al
  4052d2:	74 2a                	je     0x4052fe
  4052d4:	3c ab                	cmp    $0xab,%al
  4052d6:	74 26                	je     0x4052fe
  4052d8:	3c ac                	cmp    $0xac,%al
  4052da:	74 22                	je     0x4052fe
  4052dc:	3c ad                	cmp    $0xad,%al
  4052de:	74 1e                	je     0x4052fe
  4052e0:	3c ae                	cmp    $0xae,%al
  4052e2:	74 1a                	je     0x4052fe
  4052e4:	3c af                	cmp    $0xaf,%al
  4052e6:	74 16                	je     0x4052fe
  4052e8:	3c 6c                	cmp    $0x6c,%al
  4052ea:	74 12                	je     0x4052fe
  4052ec:	3c 6d                	cmp    $0x6d,%al
  4052ee:	74 0e                	je     0x4052fe
  4052f0:	3c 6e                	cmp    $0x6e,%al
  4052f2:	74 0a                	je     0x4052fe
  4052f4:	3c 6f                	cmp    $0x6f,%al
  4052f6:	74 06                	je     0x4052fe
  4052f8:	3c 0f                	cmp    $0xf,%al
  4052fa:	74 02                	je     0x4052fe
  4052fc:	eb 04                	jmp    0x405302
  4052fe:	c6 45 01 01          	movb   $0x1,0x1(%ebp)
  405302:	8b 45 23             	mov    0x23(%ebp),%eax
  405305:	0f b6 08             	movzbl (%eax),%ecx
  405308:	8d 04 8e             	lea    (%esi,%ecx,4),%eax
  40530b:	03 00                	add    (%eax),%eax
  40530d:	83 c0 04             	add    $0x4,%eax
  405310:	ff d0                	call   *%eax
  405312:	c6 45 01 00          	movb   $0x0,0x1(%ebp)
  405316:	c3                   	ret
  405317:	ff 45 23             	incl   0x23(%ebp)
  40531a:	fe 45 22             	incb   0x22(%ebp)
  40531d:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  405321:	75 06                	jne    0x405329
  405323:	e8 48 f9 ff ff       	call   0x404c70
  405328:	c3                   	ret
  405329:	8b 45 23             	mov    0x23(%ebp),%eax
  40532c:	0f b6 08             	movzbl (%eax),%ecx
  40532f:	8d 84 8e 00 04 00 00 	lea    0x400(%esi,%ecx,4),%eax
  405336:	03 00                	add    (%eax),%eax
  405338:	83 c0 04             	add    $0x4,%eax
  40533b:	ff d0                	call   *%eax
  40533d:	c3                   	ret
  40533e:	ff 45 23             	incl   0x23(%ebp)
  405341:	fe 45 22             	incb   0x22(%ebp)
  405344:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  405348:	75 06                	jne    0x405350
  40534a:	e8 21 f9 ff ff       	call   0x404c70
  40534f:	c3                   	ret
  405350:	8b 45 23             	mov    0x23(%ebp),%eax
  405353:	0f b6 08             	movzbl (%eax),%ecx
  405356:	8d 84 8e 00 08 00 00 	lea    0x800(%esi,%ecx,4),%eax
  40535d:	03 00                	add    (%eax),%eax
  40535f:	83 c0 04             	add    $0x4,%eax
  405362:	ff d0                	call   *%eax
  405364:	c3                   	ret
  405365:	ff 45 23             	incl   0x23(%ebp)
  405368:	fe 45 22             	incb   0x22(%ebp)
  40536b:	80 7d 22 0f          	cmpb   $0xf,0x22(%ebp)
  40536f:	75 06                	jne    0x405377
  405371:	e8 fa f8 ff ff       	call   0x404c70
  405376:	c3                   	ret
  405377:	8b 45 23             	mov    0x23(%ebp),%eax
  40537a:	0f b6 08             	movzbl (%eax),%ecx
  40537d:	8d 84 8e 00 0c 00 00 	lea    0xc00(%esi,%ecx,4),%eax
  405384:	03 00                	add    (%eax),%eax
  405386:	83 c0 04             	add    $0x4,%eax
  405389:	ff d0                	call   *%eax
  40538b:	c3                   	ret
  40538c:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  405393:	8b 45 23             	mov    0x23(%ebp),%eax
  405396:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  40539a:	3d bf 00 00 00       	cmp    $0xbf,%eax
  40539f:	7f 11                	jg     0x4053b2
  4053a1:	e8 0c f6 ff ff       	call   0x4049b2
  4053a6:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  4053aa:	7e 06                	jle    0x4053b2
  4053ac:	e8 bf f8 ff ff       	call   0x404c70
  4053b1:	c3                   	ret
  4053b2:	e8 b2 f5 ff ff       	call   0x404969
  4053b7:	8b 45 1a             	mov    0x1a(%ebp),%eax
  4053ba:	01 45 23             	add    %eax,0x23(%ebp)
  4053bd:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  4053c1:	c3                   	ret
  4053c2:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  4053c9:	8b 45 23             	mov    0x23(%ebp),%eax
  4053cc:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  4053d0:	3d bf 00 00 00       	cmp    $0xbf,%eax
  4053d5:	7f 17                	jg     0x4053ee
  4053d7:	e8 d6 f5 ff ff       	call   0x4049b2
  4053dc:	83 7d 16 01          	cmpl   $0x1,0x16(%ebp)
  4053e0:	75 69                	jne    0x40544b
  4053e2:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  4053e6:	7e 63                	jle    0x40544b
  4053e8:	e8 83 f8 ff ff       	call   0x404c70
  4053ed:	c3                   	ret
  4053ee:	3d c0 00 00 00       	cmp    $0xc0,%eax
  4053f3:	7c 56                	jl     0x40544b
  4053f5:	89 c2                	mov    %eax,%edx
  4053f7:	c1 ea 04             	shr    $0x4,%edx
  4053fa:	89 c1                	mov    %eax,%ecx
  4053fc:	83 e1 0f             	and    $0xf,%ecx
  4053ff:	83 fa 0d             	cmp    $0xd,%edx
  405402:	75 0b                	jne    0x40540f
  405404:	83 f9 00             	cmp    $0x0,%ecx
  405407:	74 42                	je     0x40544b
  405409:	e8 62 f8 ff ff       	call   0x404c70
  40540e:	c3                   	ret
  40540f:	83 fa 0e             	cmp    $0xe,%edx
  405412:	75 37                	jne    0x40544b
  405414:	83 f9 02             	cmp    $0x2,%ecx
  405417:	75 06                	jne    0x40541f
  405419:	e8 52 f8 ff ff       	call   0x404c70
  40541e:	c3                   	ret
  40541f:	83 f9 03             	cmp    $0x3,%ecx
  405422:	75 06                	jne    0x40542a
  405424:	e8 47 f8 ff ff       	call   0x404c70
  405429:	c3                   	ret
  40542a:	83 f9 06             	cmp    $0x6,%ecx
  40542d:	75 06                	jne    0x405435
  40542f:	e8 3c f8 ff ff       	call   0x404c70
  405434:	c3                   	ret
  405435:	83 f9 07             	cmp    $0x7,%ecx
  405438:	75 06                	jne    0x405440
  40543a:	e8 31 f8 ff ff       	call   0x404c70
  40543f:	c3                   	ret
  405440:	83 f9 0f             	cmp    $0xf,%ecx
  405443:	75 06                	jne    0x40544b
  405445:	e8 26 f8 ff ff       	call   0x404c70
  40544a:	c3                   	ret
  40544b:	e8 19 f5 ff ff       	call   0x404969
  405450:	8b 45 1a             	mov    0x1a(%ebp),%eax
  405453:	01 45 23             	add    %eax,0x23(%ebp)
  405456:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  40545a:	c3                   	ret
  40545b:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  405462:	8b 45 23             	mov    0x23(%ebp),%eax
  405465:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  405469:	3d bf 00 00 00       	cmp    $0xbf,%eax
  40546e:	7f 11                	jg     0x405481
  405470:	e8 3d f5 ff ff       	call   0x4049b2
  405475:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  405479:	7e 32                	jle    0x4054ad
  40547b:	e8 f0 f7 ff ff       	call   0x404c70
  405480:	c3                   	ret
  405481:	3d c0 00 00 00       	cmp    $0xc0,%eax
  405486:	7c 25                	jl     0x4054ad
  405488:	89 c2                	mov    %eax,%edx
  40548a:	c1 ea 04             	shr    $0x4,%edx
  40548d:	89 c1                	mov    %eax,%ecx
  40548f:	83 e1 0f             	and    $0xf,%ecx
  405492:	83 fa 0e             	cmp    $0xe,%edx
  405495:	75 0b                	jne    0x4054a2
  405497:	83 f9 09             	cmp    $0x9,%ecx
  40549a:	74 11                	je     0x4054ad
  40549c:	e8 cf f7 ff ff       	call   0x404c70
  4054a1:	c3                   	ret
  4054a2:	83 fa 0f             	cmp    $0xf,%edx
  4054a5:	75 06                	jne    0x4054ad
  4054a7:	e8 c4 f7 ff ff       	call   0x404c70
  4054ac:	c3                   	ret
  4054ad:	e8 b7 f4 ff ff       	call   0x404969
  4054b2:	8b 45 1a             	mov    0x1a(%ebp),%eax
  4054b5:	01 45 23             	add    %eax,0x23(%ebp)
  4054b8:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  4054bc:	c3                   	ret
  4054bd:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  4054c4:	8b 45 23             	mov    0x23(%ebp),%eax
  4054c7:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  4054cb:	3d bf 00 00 00       	cmp    $0xbf,%eax
  4054d0:	7f 1f                	jg     0x4054f1
  4054d2:	e8 db f4 ff ff       	call   0x4049b2
  4054d7:	83 7d 16 04          	cmpl   $0x4,0x16(%ebp)
  4054db:	74 0e                	je     0x4054eb
  4054dd:	83 7d 16 06          	cmpl   $0x6,0x16(%ebp)
  4054e1:	74 08                	je     0x4054eb
  4054e3:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  4054e7:	7f 02                	jg     0x4054eb
  4054e9:	eb 41                	jmp    0x40552c
  4054eb:	e8 80 f7 ff ff       	call   0x404c70
  4054f0:	c3                   	ret
  4054f1:	3d c0 00 00 00       	cmp    $0xc0,%eax
  4054f6:	7c 34                	jl     0x40552c
  4054f8:	89 c2                	mov    %eax,%edx
  4054fa:	c1 ea 04             	shr    $0x4,%edx
  4054fd:	89 c1                	mov    %eax,%ecx
  4054ff:	83 e1 0f             	and    $0xf,%ecx
  405502:	83 fa 0e             	cmp    $0xe,%edx
  405505:	75 15                	jne    0x40551c
  405507:	83 f9 08             	cmp    $0x8,%ecx
  40550a:	7d 20                	jge    0x40552c
  40550c:	83 f9 03             	cmp    $0x3,%ecx
  40550f:	74 1b                	je     0x40552c
  405511:	83 f9 02             	cmp    $0x2,%ecx
  405514:	74 16                	je     0x40552c
  405516:	e8 55 f7 ff ff       	call   0x404c70
  40551b:	c3                   	ret
  40551c:	83 fa 0f             	cmp    $0xf,%edx
  40551f:	75 0b                	jne    0x40552c
  405521:	83 f9 08             	cmp    $0x8,%ecx
  405524:	7c 06                	jl     0x40552c
  405526:	e8 45 f7 ff ff       	call   0x404c70
  40552b:	c3                   	ret
  40552c:	e8 38 f4 ff ff       	call   0x404969
  405531:	8b 45 1a             	mov    0x1a(%ebp),%eax
  405534:	01 45 23             	add    %eax,0x23(%ebp)
  405537:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  40553b:	c3                   	ret
  40553c:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  405543:	8b 45 23             	mov    0x23(%ebp),%eax
  405546:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  40554a:	3d bf 00 00 00       	cmp    $0xbf,%eax
  40554f:	7f 11                	jg     0x405562
  405551:	e8 5c f4 ff ff       	call   0x4049b2
  405556:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  40555a:	7e 22                	jle    0x40557e
  40555c:	e8 0f f7 ff ff       	call   0x404c70
  405561:	c3                   	ret
  405562:	3d c0 00 00 00       	cmp    $0xc0,%eax
  405567:	7c 15                	jl     0x40557e
  405569:	89 c2                	mov    %eax,%edx
  40556b:	c1 ea 04             	shr    $0x4,%edx
  40556e:	89 c1                	mov    %eax,%ecx
  405570:	83 e1 0f             	and    $0xf,%ecx
  405573:	83 fa 0d             	cmp    $0xd,%edx
  405576:	75 06                	jne    0x40557e
  405578:	e8 f3 f6 ff ff       	call   0x404c70
  40557d:	c3                   	ret
  40557e:	e8 e6 f3 ff ff       	call   0x404969
  405583:	8b 45 1a             	mov    0x1a(%ebp),%eax
  405586:	01 45 23             	add    %eax,0x23(%ebp)
  405589:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  40558d:	c3                   	ret
  40558e:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  405595:	8b 45 23             	mov    0x23(%ebp),%eax
  405598:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  40559c:	3d bf 00 00 00       	cmp    $0xbf,%eax
  4055a1:	7f 19                	jg     0x4055bc
  4055a3:	e8 0a f4 ff ff       	call   0x4049b2
  4055a8:	83 7d 16 05          	cmpl   $0x5,0x16(%ebp)
  4055ac:	74 08                	je     0x4055b6
  4055ae:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  4055b2:	7f 02                	jg     0x4055b6
  4055b4:	eb 32                	jmp    0x4055e8
  4055b6:	e8 b5 f6 ff ff       	call   0x404c70
  4055bb:	c3                   	ret
  4055bc:	3d c0 00 00 00       	cmp    $0xc0,%eax
  4055c1:	7c 25                	jl     0x4055e8
  4055c3:	89 c2                	mov    %eax,%edx
  4055c5:	c1 ea 04             	shr    $0x4,%edx
  4055c8:	89 c1                	mov    %eax,%ecx
  4055ca:	83 e1 0f             	and    $0xf,%ecx
  4055cd:	83 fa 0c             	cmp    $0xc,%edx
  4055d0:	75 0b                	jne    0x4055dd
  4055d2:	83 f9 08             	cmp    $0x8,%ecx
  4055d5:	7c 11                	jl     0x4055e8
  4055d7:	e8 94 f6 ff ff       	call   0x404c70
  4055dc:	c3                   	ret
  4055dd:	83 fa 0f             	cmp    $0xf,%edx
  4055e0:	75 06                	jne    0x4055e8
  4055e2:	e8 89 f6 ff ff       	call   0x404c70
  4055e7:	c3                   	ret
  4055e8:	e8 7c f3 ff ff       	call   0x404969
  4055ed:	8b 45 1a             	mov    0x1a(%ebp),%eax
  4055f0:	01 45 23             	add    %eax,0x23(%ebp)
  4055f3:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  4055f7:	c3                   	ret
  4055f8:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  4055ff:	8b 45 23             	mov    0x23(%ebp),%eax
  405602:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  405606:	3d bf 00 00 00       	cmp    $0xbf,%eax
  40560b:	7f 11                	jg     0x40561e
  40560d:	e8 a0 f3 ff ff       	call   0x4049b2
  405612:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  405616:	7e 27                	jle    0x40563f
  405618:	e8 53 f6 ff ff       	call   0x404c70
  40561d:	c3                   	ret
  40561e:	3d c0 00 00 00       	cmp    $0xc0,%eax
  405623:	7c 1a                	jl     0x40563f
  405625:	89 c2                	mov    %eax,%edx
  405627:	c1 ea 04             	shr    $0x4,%edx
  40562a:	89 c1                	mov    %eax,%ecx
  40562c:	83 e1 0f             	and    $0xf,%ecx
  40562f:	83 fa 0d             	cmp    $0xd,%edx
  405632:	75 0b                	jne    0x40563f
  405634:	83 f9 09             	cmp    $0x9,%ecx
  405637:	74 06                	je     0x40563f
  405639:	e8 32 f6 ff ff       	call   0x404c70
  40563e:	c3                   	ret
  40563f:	e8 25 f3 ff ff       	call   0x404969
  405644:	8b 45 1a             	mov    0x1a(%ebp),%eax
  405647:	01 45 23             	add    %eax,0x23(%ebp)
  40564a:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  40564e:	c3                   	ret
  40564f:	c7 45 1a 00 00 00 00 	movl   $0x0,0x1a(%ebp)
  405656:	8b 45 23             	mov    0x23(%ebp),%eax
  405659:	0f b6 40 01          	movzbl 0x1(%eax),%eax
  40565d:	3d bf 00 00 00       	cmp    $0xbf,%eax
  405662:	7f 11                	jg     0x405675
  405664:	e8 49 f3 ff ff       	call   0x4049b2
  405669:	83 7d 16 07          	cmpl   $0x7,0x16(%ebp)
  40566d:	7e 52                	jle    0x4056c1
  40566f:	e8 fc f5 ff ff       	call   0x404c70
  405674:	c3                   	ret
  405675:	3d c0 00 00 00       	cmp    $0xc0,%eax
  40567a:	7c 45                	jl     0x4056c1
  40567c:	89 c2                	mov    %eax,%edx
  40567e:	c1 ea 04             	shr    $0x4,%edx
  405681:	89 c1                	mov    %eax,%ecx
  405683:	83 e1 0f             	and    $0xf,%ecx
  405686:	83 fa 0c             	cmp    $0xc,%edx
  405689:	75 06                	jne    0x405691
  40568b:	e8 e0 f5 ff ff       	call   0x404c70
  405690:	c3                   	ret
  405691:	83 fa 0d             	cmp    $0xd,%edx
  405694:	75 06                	jne    0x40569c
  405696:	e8 d5 f5 ff ff       	call   0x404c70
  40569b:	c3                   	ret
  40569c:	83 fa 0e             	cmp    $0xe,%edx
  40569f:	75 10                	jne    0x4056b1
  4056a1:	83 f9 00             	cmp    $0x0,%ecx
  4056a4:	74 1b                	je     0x4056c1
  4056a6:	83 f9 08             	cmp    $0x8,%ecx
  4056a9:	7d 16                	jge    0x4056c1
  4056ab:	e8 c0 f5 ff ff       	call   0x404c70
  4056b0:	c3                   	ret
  4056b1:	83 fa 0f             	cmp    $0xf,%edx
  4056b4:	75 0b                	jne    0x4056c1
  4056b6:	83 f9 08             	cmp    $0x8,%ecx
  4056b9:	7c 06                	jl     0x4056c1
  4056bb:	e8 b0 f5 ff ff       	call   0x404c70
  4056c0:	c3                   	ret
  4056c1:	e8 a3 f2 ff ff       	call   0x404969
  4056c6:	8b 45 1a             	mov    0x1a(%ebp),%eax
  4056c9:	01 45 23             	add    %eax,0x23(%ebp)
  4056cc:	83 45 23 02          	addl   $0x2,0x23(%ebp)
  4056d0:	c3                   	ret
